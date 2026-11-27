"""Persist submitted extractions and run one job at a time."""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from uuid import uuid4

from .agent import PaperProgress, load_paper_with_timing, run_agent, run_directory

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    prompt TEXT NOT NULL,
    has_si INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS job_papers (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    relative_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT,
    completed INTEGER,
    total INTEGER,
    preview_json TEXT,
    result_path TEXT,
    error TEXT,
    UNIQUE (job_id, relative_path)
);
"""


class ExtractionJobs:
    def __init__(self, data_dir: Path, workers: int):
        self.data_dir = data_dir
        self.workers = workers
        self.database = data_dir / "jobs.sqlite3"
        self.wakeup = Event()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.database)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            yield db

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            error = "The service stopped before this extraction finished."
            db.execute("""
                UPDATE job_papers SET status = 'interrupted', stage = NULL, error = ?
                WHERE status IN ('queued', 'running')
                AND job_id IN (SELECT id FROM jobs WHERE status = 'running')
            """, (error,))
            db.execute("""
                UPDATE jobs SET status = 'interrupted', stage = NULL, finished_at = ?, error = ?
                WHERE status = 'running'
            """, (datetime.now(UTC).isoformat(), error))

    def directory(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def enqueue(self, job_id: str, mode: str, prompt: str, paths: list[str], has_si: bool) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO jobs (id, mode, prompt, has_si, created_at) VALUES (?, ?, ?, ?, ?)",
                       (job_id, mode, prompt, has_si, datetime.now(UTC).isoformat()))
            db.executemany("INSERT INTO job_papers (id, job_id, relative_path) VALUES (?, ?, ?)",
                           [(uuid4().hex, job_id, path) for path in paths])
        self.wakeup.set()

    def snapshot(self, job_id: str) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
        with self.connect() as db:
            db.execute("BEGIN")
            job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            papers = db.execute("SELECT * FROM job_papers WHERE job_id = ? ORDER BY relative_path",
                                (job_id,)).fetchall()
        return job, papers

    def _claim(self) -> sqlite3.Row | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1").fetchone()
            if job:
                db.execute("""
                    UPDATE jobs SET status = 'running', stage = 'processing', started_at = ? WHERE id = ?
                """, (datetime.now(UTC).isoformat(), job["id"]))
        return job

    def _paper_progress(self, job: sqlite3.Row, paper_id: str, progress: PaperProgress) -> None:
        job_dir = self.directory(job["id"])
        stage = progress.stage
        status = stage if stage in {"succeeded", "failed"} else "running"
        result = progress.result_path.relative_to(job_dir).as_posix() if progress.result_path else None
        preview = json.dumps(progress.rows, ensure_ascii=False) if progress.rows is not None else None
        error = progress.error.replace(str(job_dir), ".") if progress.error else None
        with self.connect() as db:
            db.execute("""
                UPDATE job_papers SET status = ?, stage = ?,
                    completed = COALESCE(?, completed), total = COALESCE(?, total),
                    preview_json = CASE WHEN ? IS NOT NULL THEN NULL ELSE COALESCE(?, preview_json) END,
                    result_path = ?, error = ?
                WHERE id = ?
            """, (status, stage if status == "running" else None, progress.completed, progress.total,
                  result, preview, result, error, paper_id))
            if job["mode"] == "single" and status == "running":
                db.execute("UPDATE jobs SET stage = ? WHERE id = ?", (stage, job["id"]))

    def _summarizing(self, job_id: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET stage = 'summarizing' WHERE id = ?", (job_id,))

    def _finish(self, job_id: str, status: str, error: str | None = None) -> None:
        with self.connect() as db:
            if status == "failed":
                db.execute("""
                    UPDATE job_papers SET status = 'failed', stage = NULL, error = ?
                    WHERE job_id = ? AND status IN ('queued', 'running')
                """, (error, job_id))
            db.execute("""
                UPDATE jobs SET status = ?, stage = NULL, finished_at = ?, error = ? WHERE id = ?
            """, (status, datetime.now(UTC).isoformat(), error, job_id))

    def _run(self, job: sqlite3.Row) -> None:
        job_id = job["id"]
        job_dir = self.directory(job_id)
        input_dir, output_dir = job_dir / "input", job_dir / "output"
        _, papers = self.snapshot(job_id)
        if job["mode"] == "single":
            paper = papers[0]

            def report(progress: PaperProgress) -> None:
                self._paper_progress(job, paper["id"], progress)

            report(PaperProgress("reading"))
            si_path = str(job_dir / "supporting.pdf") if job["has_si"] else None
            try:
                text, name, conversion_time = load_paper_with_timing(
                    str(input_dir / paper["relative_path"]), si_path, save_text=False,
                )
                output_prefix = output_dir / name
                run_agent(text, name, job["prompt"], output_prefix=output_prefix,
                          pdf_conversion_elapsed=conversion_time, on_progress=report)
            except sqlite3.Error:
                raise
            except Exception as exc:
                log.exception("Extraction failed for paper %s", paper["id"])
                report(PaperProgress("failed", error=str(exc)))
                self._finish(job_id, "failed", "Paper extraction failed.")
                return
            report(PaperProgress("succeeded", result_path=output_dir / f"{name}_extraction.json"))
            self._finish(job_id, "succeeded")
            return

        paper_ids = {paper["relative_path"].casefold(): paper["id"] for paper in papers}

        def report_batch(path: str, progress: PaperProgress) -> None:
            self._paper_progress(job, paper_ids[path.casefold()], progress)

        succeeded, failed = run_directory(input_dir, job["prompt"], output_dir, self.workers,
                                          on_paper_progress=report_batch,
                                          on_summarizing=lambda: self._summarizing(job_id))
        if not succeeded:
            self._finish(job_id, "failed", "All papers failed.")
        else:
            self._finish(job_id, "partial_success" if failed else "succeeded")

    def work(self, stop: Event) -> None:
        while not stop.is_set():
            self.wakeup.clear()
            try:
                job = self._claim()
                if job is None:
                    self.wakeup.wait(1)
                    continue
                try:
                    self._run(job)
                except (Exception, SystemExit):
                    log.exception("Extraction job %s failed", job["id"])
                    self._finish(job["id"], "failed", "The extraction could not finish.")
            except sqlite3.Error:
                log.exception("Could not update the extraction queue")
                stop.wait(1)

"""HTTP interface for submitting and polling paper extractions."""

import json
import os
import re
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from shutil import copyfileobj, rmtree
from threading import Event, Thread
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .config import DEFAULT_PROMPT
from .jobs import ExtractionJobs

Mode = Literal["single", "directory"]
PaperState = Literal["queued", "running", "succeeded", "failed", "interrupted"]
JobState = Literal["queued", "running", "succeeded", "partial_success", "failed", "interrupted"]
router = APIRouter(prefix="/api/extractions")


class SubmittedJob(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class PaperStatus(BaseModel):
    paper_id: str
    path: str
    status: PaperState
    stage: str | None
    parameters_completed: int | None
    parameters_total: int | None
    error: str | None


class JobProgress(BaseModel):
    total: int
    succeeded: int
    failed: int
    interrupted: int


class PaperResult(BaseModel):
    paper_id: str
    partial: bool
    doi: str | None = None
    results: list[dict]
    metrics: dict | None = None


class JobStatus(BaseModel):
    job_id: str
    mode: Mode
    status: JobState
    stage: str | None
    progress: JobProgress
    papers: list[PaperStatus]
    paper: PaperResult | None = None
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


def _upload_path(value: str) -> PurePosixPath:
    for part in value.split("/"):
        if (not part or part in {".", ".."} or part.endswith((".", " "))
                or re.search(r'[<>:"\\|?*\x00-\x1f]', part)
                or re.fullmatch(r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³]",
                                part.split(".", 1)[0].rstrip().upper())):
            raise HTTPException(422, "Files must have valid relative paths.")
    path = PurePosixPath(value)
    if path.suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(422, "Only PDF and TXT papers are supported.")
    return path


def _save_file(upload: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as out:
        copyfileobj(upload.file, out)


@router.post("", status_code=202)
def submit_extraction(
    request: Request, response: Response,
    files: Annotated[list[UploadFile], File()],
    mode: Annotated[Mode, Form()] = "single",
    prompt: Annotated[str, Form()] = DEFAULT_PROMPT,
    relative_paths: Annotated[list[str] | None, Form()] = None,
    si: Annotated[UploadFile | None, File()] = None,
) -> SubmittedJob:
    if mode == "single":
        if len(files) != 1 or relative_paths is not None:
            raise HTTPException(422, "Single mode expects one file and no relative_paths.")
        path = _upload_path(files[0].filename or "")
        if len(path.parts) != 1:
            raise HTTPException(422, "Single mode expects a filename.")
        paths = [path]
        if si is not None and (path.suffix.lower() != ".pdf"
                               or PurePosixPath(si.filename or "").suffix.lower() != ".pdf"):
            raise HTTPException(422, "Supporting information requires a PDF paper and a PDF supplement.")
    else:
        if si is not None or not relative_paths or len(relative_paths) != len(files):
            raise HTTPException(422, "Directory mode expects one relative path per file and no SI upload.")
        paths = [_upload_path(path) for path in relative_paths]
        keys = {path.as_posix().casefold() for path in paths}
        if len(keys) != len(paths) or any(
            parent.as_posix().casefold() in keys for path in paths for parent in path.parents
        ):
            raise HTTPException(422, "File paths must be unique and cannot also be directories.")

    jobs: ExtractionJobs = request.app.state.jobs
    job_id = uuid4().hex
    status_url = request.url_for("get_extraction", job_id=job_id).path
    job_dir = jobs.directory(job_id)
    job_dir.mkdir(parents=True)
    try:
        for upload, path in zip(files, paths, strict=True):
            _save_file(upload, job_dir / "input" / path)
        if si is not None:
            _save_file(si, job_dir / "supporting.pdf")
        jobs.enqueue(job_id, mode, prompt.strip() or DEFAULT_PROMPT, [path.as_posix() for path in paths], si is not None)
    except Exception:
        rmtree(job_dir, ignore_errors=True)
        raise

    response.headers["Location"] = status_url
    return SubmittedJob(job_id=job_id, status_url=status_url)


@router.get("/{job_id}")
def get_extraction(request: Request, response: Response, job_id: str, paper_id: str | None = None) -> JobStatus:
    jobs: ExtractionJobs = request.app.state.jobs
    job, papers = jobs.snapshot(job_id)
    if job is None:
        raise HTTPException(404, "Extraction job not found.")
    response.headers["Cache-Control"] = "no-store"

    paper_result = None
    if paper_id is not None or job["mode"] == "single":
        paper = next((paper for paper in papers if paper_id is None or paper["id"] == paper_id), None)
        if paper is None:
            raise HTTPException(404, "Paper not found in this job.")
        complete = paper["result_path"] is not None
        doi = metrics = None
        if complete:
            path = jobs.directory(job_id) / paper["result_path"]
            try:
                output = json.loads(path.read_text(encoding="utf-8"))
                trace = json.loads(path.with_name(f"{path.stem}_trace.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise HTTPException(500, "Stored extraction results are unavailable.") from exc
            rows, doi, metrics = output["results"], output.get("doi"), trace["metrics"]
        else:
            rows = json.loads(paper["preview_json"]) if paper["preview_json"] else []
        paper_result = PaperResult(paper_id=paper["id"], partial=not complete, doi=doi, metrics=metrics,
                                   results=[{**row, "row_id": index} for index, row in enumerate(rows, 1)])

    states = Counter(paper["status"] for paper in papers)
    return JobStatus(
        job_id=job_id, mode=job["mode"], status=job["status"], stage=job["stage"], error=job["error"],
        created_at=job["created_at"], started_at=job["started_at"], finished_at=job["finished_at"],
        progress=JobProgress(total=len(papers), succeeded=states["succeeded"], failed=states["failed"],
                             interrupted=states["interrupted"]),
        papers=[
            PaperStatus(paper_id=paper["id"], path=paper["relative_path"], status=paper["status"],
                        stage=paper["stage"], parameters_completed=paper["completed"],
                        parameters_total=paper["total"], error=paper["error"])
            for paper in papers
        ],
        paper=paper_result,
    )


def create_app(data_dir: Path | None = None, workers: int | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        root = Path(__file__).resolve().parents[2]
        default_dir = (root if (root / "pyproject.toml").is_file() else Path.cwd()) / ".photocatevo"
        directory = data_dir if data_dir is not None else Path(os.getenv("PHOTOCATEVO_DATA_DIR", str(default_dir)))
        concurrency = workers if workers is not None else int(os.getenv("PHOTOCATEVO_WORKERS", "1"))
        if concurrency < 1:
            raise ValueError("PHOTOCATEVO_WORKERS must be at least 1.")
        jobs = ExtractionJobs(directory.resolve(), concurrency)
        jobs.initialize()
        app.state.jobs = jobs
        stop = Event()
        worker = Thread(target=jobs.work, args=(stop,), name="photocatevo-dispatcher")
        worker.start()
        try:
            yield
        finally:
            stop.set()
            jobs.wakeup.set()
            await run_in_threadpool(worker.join)

    app = FastAPI(title="PhotoCatEvo", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)
    return app


app = create_app()

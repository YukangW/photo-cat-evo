import json
import time
from threading import Event

import pytest

from PhotoCatEvo import agent, pdf_to_text

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from PhotoCatEvo.api import create_app
from PhotoCatEvo.jobs import ExtractionJobs
from PhotoCatEvo.llm import model_settings


def test_health(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def _wait_for(client, url, ready):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        state = response.json()
        if ready(state):
            return state
        time.sleep(0.02)
    pytest.fail(f"Expected progress was not reached. Last response: {state}")


@pytest.fixture
def extraction(monkeypatch):
    def configure(names, before_extract):
        monkeypatch.setattr(agent, "plan_targets", lambda *a, **kw: ([{"name": name} for name in names], []))

        def extract(text, target, *args):
            before_extract(text, target["name"])
            result = {"sample_name": "CN-550", "value": 2.65, "unit": "eV",
                      "evidence": "CN-550 has a band gap of 2.65 eV", "status": "verified"}
            return result, [], {"status": "verified", "elapsed_seconds": 0,
                                "token_usage": agent.TokenUsage().as_dict()}

        monkeypatch.setattr(agent, "_extract_with_metrics", extract)

    return configure


def test_single_returns_preview_then_standardized_result(tmp_path, monkeypatch, extraction):
    release = Event()

    def before_extract(text, name):
        if name == "second item":
            assert release.wait(5)

    def convert(paper, si):
        assert si.endswith("supporting.pdf")
        return "CN-550 has a band gap of 2.65 eV", 0

    extraction(["band gap", "second item"], before_extract)
    monkeypatch.setattr(pdf_to_text, "convert", convert)
    with TestClient(create_app(tmp_path)) as client:
        default_prompt = client.get("/api/extractions/defaults").json()["prompt"]
        assert default_prompt == "Find the band gap and BET surface area."
        response = client.post("/api/extractions", data={"prompt": " "}, files=[
            ("files", ("paper.pdf", b"mock PDF")), ("si", ("supplement.pdf", b"mock SI")),
        ])
        assert response.status_code == 202
        url = response.headers["location"]
        assert response.json()["status_url"] == url
        try:
            state = _wait_for(client, url, lambda state: state["papers"][0]["parameters_completed"] == 1)
            assert state["papers"][0]["current_parameter"] == "second item"
            preview = state["paper"]
            assert preview["partial"] and preview["results"][0]["evidence"]
            assert client.get(f"{url}/all_samples.csv").status_code == 409
        finally:
            release.set()
        finished = _wait_for(client, url, lambda state: state["status"] == "succeeded")
        assert finished["progress"]["succeeded"] == 1
        final = finished["paper"]
        assert final["partial"] is False
        assert final["results"][0]["row_id"] == preview["results"][0]["row_id"]
        assert final["results"][0]["standard_value"] == 2.65
        assert final["metrics"]["parameters_attempted"] == 2
        output = tmp_path / "jobs" / finished["job_id"] / "output" / "paper_extraction.json"
        assert json.loads(output.read_text(encoding="utf-8"))["prompt"] == default_prompt
        summary = client.get(f"{url}/all_samples.csv")
        assert summary.status_code == 200
        assert summary.headers["content-disposition"] == 'attachment; filename="all_samples.csv"'
        assert summary.content.decode("utf-8-sig").splitlines()[0].startswith("paper,doi,sample_name,band gap")
        assert "CN-550,2.65,eV" in summary.text


def test_directory_publishes_results_during_parallel_extraction(tmp_path, extraction):
    slow_started, release = Event(), Event()

    def before_extract(text, name):
        settings = model_settings.get()
        assert settings.base_url == "http://127.0.0.1:9000/v1"
        assert settings.model == "custom-model"
        if text == "slow":
            slow_started.set()
            assert release.wait(5)
        else:
            assert slow_started.wait(5)

    extraction(["band gap"], before_extract)
    with TestClient(create_app(tmp_path, workers=1)) as client:
        assert client.get("/api/extractions/defaults").json()["workers"] == 1
        response = client.post("/api/extractions", data={
            "mode": "directory", "relative_paths": ["batch/a.txt", "batch/sub/b.txt", "batch/c.txt"],
            "base_url": "http://127.0.0.1:9000/v1", "model": "custom-model", "workers": 2,
        }, files=[
            ("files", ("a.txt", b"fast")), ("files", ("b.txt", b"slow")), ("files", ("c.txt", b"\xff")),
        ])
        assert response.status_code == 202
        url = response.json()["status_url"]
        try:
            state = _wait_for(client, url, lambda state: state["progress"]["succeeded"] == 1)
            assert state["status"] == "running"
            assert state["paper"] is None
            paper = next(paper for paper in state["papers"] if paper["status"] == "succeeded")
            assert paper["path"] == "batch/a.txt"
            slow = next(paper for paper in state["papers"] if paper["path"] == "batch/sub/b.txt")
            assert slow["stage"] == "extracting"
            assert slow["current_parameter"] == "band gap"
            assert (slow["parameters_completed"], slow["parameters_total"]) == (0, 1)
            detail = client.get(url, params={"paper_id": paper["paper_id"]}).json()
            assert detail["paper"]["results"][0]["value"] == 2.65
        finally:
            release.set()
        finished = _wait_for(client, url, lambda state: state["status"] == "partial_success")
        assert finished["progress"] == {"total": 3, "succeeded": 2, "failed": 1, "interrupted": 0}
        assert all(paper["current_parameter"] is None for paper in finished["papers"])
        assert next(paper for paper in finished["papers"] if paper["status"] == "failed")["error"]
        assert client.get(url, params={"paper_id": "missing"}).status_code == 404
        output = tmp_path / "jobs" / finished["job_id"] / "output"
        summary = client.get(f"{url}/all_samples.csv")
        assert summary.status_code == 200
        assert summary.content == (output / "all_samples.csv").read_bytes()
        assert (output / "cost_metrics.csv").is_file()


def test_unsafe_upload_is_rejected_and_startup_recovers_queue(tmp_path, extraction):
    jobs = ExtractionJobs(tmp_path, 1)
    jobs.initialize()
    jobs.enqueue("stopped", "single", "band gap", ["a.txt"], False)
    jobs._claim()
    jobs.enqueue("waiting", "single", "band gap", ["b.txt"], False)
    paper = jobs.directory("waiting") / "input" / "b.txt"
    paper.parent.mkdir(parents=True)
    paper.write_text("CN-550 has a band gap of 2.65 eV", encoding="utf-8")

    extraction(["band gap"], lambda *args: None)
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/extractions", data={"mode": "directory", "relative_paths": "../escape.txt"},
                               files={"files": ("a.txt", b"data")})
        assert response.status_code == 422
        stopped = client.get("/api/extractions/stopped").json()
        assert stopped["status"] == "interrupted"
        assert stopped["progress"]["interrupted"] == 1
        _wait_for(client, "/api/extractions/waiting", lambda state: state["status"] == "succeeded")

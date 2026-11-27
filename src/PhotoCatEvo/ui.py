"""Gradio interface for the paper extraction API."""

import os
import time
from collections.abc import Iterator
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr
import requests

POLL_INTERVAL = 1
DEFAULT_THREADS = 1
COMPLETED_STATES = {"succeeded", "partial_success"}
TERMINAL_STATES = COMPLETED_STATES | {"failed", "interrupted"}
MODEL_PRESETS = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
}
STAGE_LABELS = {
    "queued": "Queued",
    "processing": "Processing papers",
    "reading": "Parsing paper",
    "planning": "Planning extraction",
    "extracting": "Extracting parameters",
    "standardizing": "Standardizing units",
    "summarizing": "Preparing the sample summary",
    "succeeded": "Extraction complete",
    "partial_success": "Extraction partially complete",
    "failed": "Extraction failed",
    "interrupted": "Extraction interrupted",
}
RESULT_LABELS = {
    "verified": "Verified",
    "range_only": "Range only",
    "not_reported": "Not reported",
    "failed": "Extraction failed",
}

# Gradio keeps uploaded files but drops the relative paths from the browser's folder picker.
CAPTURE_FOLDER_PATHS = """
() => {
    document.addEventListener("change", event => {
        const input = event.composedPath()[0];
        if (input.matches?.("input[webkitdirectory]")) {
            window.photoCatEvoFolderPaths = Array.from(input.files, file => file.webkitRelativePath);
        }
    }, true);
    return [];
}
"""


def _load_defaults(api_url: str) -> tuple[str, str | None, str]:
    try:
        response = requests.get(f"{api_url}/api/extractions/defaults", timeout=(3, 5))
        response.raise_for_status()
        defaults = response.json()
        return defaults["base_url"], defaults["model"], defaults.get("prompt", "")
    except requests.RequestException:
        return "", None, ""


def _model_id(model: str, base_url: str) -> str:
    if model not in MODEL_PRESETS:
        return model
    if base_url.lower().startswith("https://openrouter.ai/"):
        return f"qwen/{model}"
    return MODEL_PRESETS[model]


def _measurement(row: dict, prefix: str = "") -> str:
    value = row.get(prefix + "value")
    bounds = row.get(prefix + "range")
    uncertainty = row.get(prefix + "error")
    unit = row.get(prefix + "unit")
    if value is None or value == "":
        measurement = str(bounds) if bounds else ""
    else:
        measurement = str(value)
        if uncertainty:
            measurement += f" {uncertainty}"
        if bounds:
            measurement += f" ({bounds})"
    return f"{measurement} {unit}" if measurement and unit else measurement


def _results(state: dict, paper_path: str) -> list[list[str]]:
    paper = state.get("paper") or {}
    rows = []
    for result in paper.get("results", []):
        status = result["status"]
        standardized = _measurement(result, "standard_")
        if paper["partial"] and state["status"] == "running" and status in {"verified", "range_only"}:
            standardized = standardized or "Pending"
        evidence = "\n".join(dict.fromkeys(
            result[key] for key in ("evidence", "range_evidence", "error_evidence") if result.get(key)
        ))
        rows.append([
            paper_path, result["item"], result.get("sample_name") or "",
            _measurement(result), standardized, RESULT_LABELS.get(status, status), evidence,
        ])
    return rows


def _paper_stage_text(paper: dict) -> str:
    stage = paper["stage"] or paper["status"]
    if stage == "extracting" and (parameter := paper.get("current_parameter")):
        return f'Extracting {parameter} ({paper["parameters_completed"] + 1}/{paper["parameters_total"]})'
    return STAGE_LABELS.get(stage, stage)


def _status_text(state: dict) -> str:
    stage = state["stage"] or state["status"]
    lines = [STAGE_LABELS.get(stage, stage)]
    if state["mode"] == "single":
        paper = state["papers"][0]
        if paper.get("current_parameter"):
            lines[0] = _paper_stage_text(paper)
        if paper["parameters_total"] is not None:
            lines.append(f'Parameters processed: {paper["parameters_completed"]} / {paper["parameters_total"]}')
        if error := paper["error"] or state["error"]:
            lines.append(error)
    else:
        progress = state["progress"]
        done = progress["succeeded"] + progress["failed"] + progress["interrupted"]
        active = sum(paper["status"] == "running" for paper in state["papers"])
        lines.append(f'Documents processed: {done} / {progress["total"]} · {active} active')
        for paper in sorted(state["papers"], key=lambda paper: paper["status"] != "running"):
            line = f'{paper["path"]} — {_paper_stage_text(paper)}'
            if paper["error"]:
                line += f' — {paper["error"]}'
            lines.append(line)
        if state["error"]:
            lines.append(state["error"])
    return "\n".join(lines)


def extract_papers(
    api_url: str, paper_paths: list[str] | None, prompt: str,
    base_url: str | None, model: str | None, workers: int | None,
    folder_files: list[str] | None = None, folder_paths: list[str] | None = None,
) -> Iterator[tuple[str, list[list[str]], dict, dict]]:
    rows = []
    ready = gr.update(value="Extract", interactive=True)
    clear_download = gr.update(value=None, interactive=False)
    if folder_paths is not None:
        if len(folder_files or []) != len(folder_paths):
            yield "Choose the folder again to upload its files.", rows, ready, clear_download
            return
        files_and_paths = [
            (Path(file), relative_path) for file, relative_path in zip(folder_files or [], folder_paths, strict=True)
            if Path(relative_path).suffix.lower() in {".pdf", ".txt"}
        ]
    else:
        files_and_paths = [(Path(path), Path(path).name) for path in paper_paths or []]
    if not files_and_paths:
        yield "Upload at least one PDF or TXT file to start.", rows, ready, clear_download
        return

    yield "Submitting papers…", rows, gr.update(value="Extracting…", interactive=False), clear_download
    data = {"prompt": prompt or "", "workers": str(workers if workers is not None else DEFAULT_THREADS)}
    base_url = (base_url or "").strip()
    if base_url:
        data["base_url"] = base_url
    if model:
        data["model"] = _model_id(model.strip(), base_url)
    if folder_paths is not None or len(files_and_paths) > 1:
        data.update(mode="directory", relative_paths=[relative_path for _, relative_path in files_and_paths])

    completed_papers = {}
    try:
        with requests.Session() as session:
            with ExitStack() as uploads:
                files = [("files", (paper.name, uploads.enter_context(paper.open("rb")))) for paper, _ in files_and_paths]
                response = session.post(f"{api_url}/api/extractions", files=files, data=data, timeout=(5, 60))
            response.raise_for_status()
            status_url = api_url + response.json()["status_url"]

            while True:
                response = session.get(status_url, timeout=(5, 10))
                response.raise_for_status()
                state = response.json()
                if state["mode"] == "single":
                    rows = _results(state, state["papers"][0]["path"])
                else:
                    for paper in state["papers"]:
                        paper_id = paper["paper_id"]
                        if paper["status"] == "succeeded" and paper_id not in completed_papers:
                            detail = session.get(status_url, params={"paper_id": paper_id}, timeout=(5, 10))
                            detail.raise_for_status()
                            completed_papers[paper_id] = _results(detail.json(), paper["path"])
                    rows = [row for paper in state["papers"] for row in completed_papers.get(paper["paper_id"], [])]
                status_text = _status_text(state)
                status = state["status"]
                if status in COMPLETED_STATES:
                    try:
                        csv = session.get(f"{status_url}/all_samples.csv", timeout=(5, 30))
                        csv.raise_for_status()
                        with TemporaryDirectory() as directory:
                            path = Path(directory) / "all_samples.csv"
                            path.write_bytes(csv.content)
                            yield status_text, rows, ready, gr.update(value=path, interactive=True)
                    except (requests.RequestException, OSError):
                        yield f"{status_text}\nCould not prepare the CSV download.", rows, ready, clear_download
                    return
                if status in TERMINAL_STATES:
                    yield status_text, rows, ready, clear_download
                    return
                yield status_text, rows, gr.skip(), gr.skip()
                time.sleep(POLL_INTERVAL)
    except requests.HTTPError as exc:
        yield f"The extraction API returned HTTP {exc.response.status_code}.", rows, ready, clear_download
    except requests.RequestException:
        yield "Could not read a response from the extraction API. Check that FastAPI is running.", rows, ready, clear_download
    except OSError:
        yield "Could not read the uploaded file.", rows, ready, clear_download


def create_ui(api_url: str) -> gr.Blocks:
    api_url = api_url.rstrip("/")
    with gr.Blocks(title="PhotoCatEvo", analytics_enabled=False) as app:
        gr.Markdown("# PhotoCatEvo")
        with gr.Row():
            with gr.Column():
                with gr.Group(elem_id="paper-uploads"):
                    papers = gr.File(label="Papers", file_count="multiple", file_types=[".pdf", ".txt"],
                                     type="filepath", height=110, elem_id="paper-upload")
                    folder = gr.UploadButton(file_count="directory", visible="hidden", elem_id="folder-upload")
                    gr.HTML(
                        html_template="""
                            <div class="upload-choices" role="group" aria-label="Upload options" hidden>
                                <button type="button" data-picker="files">Choose files</button>
                                <button type="button" data-picker="folder">Choose folder</button>
                            </div>
                        """,
                        css_template="""
                            .upload-choices:not([hidden]) { display: flex; gap: 8px; padding: 8px; }
                            .upload-choices button {
                                flex: 1;
                                padding: 6px 10px;
                                border-radius: var(--radius-md);
                                background: var(--button-secondary-background-fill);
                                color: var(--button-secondary-text-color);
                            }
                        """,
                        js_on_load="""
                            const choices = element.querySelector(".upload-choices");
                            let opener;
                            document.addEventListener("click", event => {
                                const target = event.composedPath()[0];
                                if (target.matches?.("input[type=file]")) return;
                                const button = event.composedPath().find(node =>
                                    node.matches?.("#paper-upload button") && node.querySelector("input[type=file]")
                                );
                                if (!button) return;
                                event.preventDefault();
                                event.stopPropagation();
                                opener = button;
                                choices.hidden = !choices.hidden;
                                if (!choices.hidden) choices.querySelector("button").focus();
                            }, true);
                            choices.addEventListener("click", event => {
                                const picker = event.target.closest("[data-picker]")?.dataset.picker;
                                if (!picker) return;
                                choices.hidden = true;
                                if (picker === "folder") {
                                    document.getElementById("folder-upload").click();
                                } else {
                                    const input = opener.querySelector("input[type=file]");
                                    input.value = "";
                                    input.click();
                                }
                            });
                            choices.addEventListener("keydown", event => {
                                if (event.key === "Escape") {
                                    choices.hidden = true;
                                    opener.focus();
                                }
                            });
                        """,
                    )
                folder_paths = gr.JSON(visible=False)
                prompt = gr.Textbox(label="Extraction prompt", lines=2)
                with gr.Row():
                    submit = gr.Button("Extract", variant="primary")
                    download = gr.DownloadButton("Download CSV", interactive=False)
            with gr.Column():
                base_url = gr.Textbox(label="Model API Base URL", placeholder="Backend default")
                with gr.Row():
                    model = gr.Dropdown(choices=list(MODEL_PRESETS), label="Model",
                                        allow_custom_value=True, value=None)
                    workers = gr.Number(value=DEFAULT_THREADS, label="Extraction threads", precision=0, minimum=1)
                status = gr.Textbox(label="Status", value="Ready to upload", lines=3, max_lines=12,
                                    autoscroll=False, interactive=False)
        results = gr.Dataframe(
            label="Extraction results", interactive=False, wrap=True,
            headers=["Paper", "Parameter", "Sample", "Reported", "Standardized", "Status", "Evidence"],
            datatype="str", column_widths=[140, 145, 110, 115, 115, 120, 305],
        )
        app.load(partial(_load_defaults, api_url), outputs=[base_url, model, prompt], js=CAPTURE_FOLDER_PATHS,
                 queue=False, show_progress="hidden", api_visibility="private")
        folder.upload(fn=None, outputs=[folder_paths, papers, status], queue=False, js="""
            () => {
                const paths = window.photoCatEvoFolderPaths ?? [];
                const name = paths[0]?.split("/")[0] ?? "";
                return [paths, null, name ? "Folder selected: " + name : "Choose a folder"];
            }
        """)
        papers.upload(fn=None, outputs=[folder_paths, status], queue=False,
                      js='() => [null, "Files selected"]')
        submit.click(partial(extract_papers, api_url), inputs=[papers, prompt, base_url, model, workers, folder, folder_paths],
                     outputs=[status, results, submit, download], api_name="extract",
                     trigger_mode="once", show_progress="minimal")
    return app.queue()


def main() -> None:
    api_url = os.getenv("PHOTOCATEVO_API_URL", "http://127.0.0.1:8001")
    create_ui(api_url).launch(server_name="127.0.0.1", server_port=7860,
                              theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"))


if __name__ == "__main__":
    main()

"""FastAPI product surface for the research-agent demonstration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from threading import Lock, Thread
from typing import Any, Literal, Mapping
from urllib.parse import unquote
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

from .agent import ResearchAgent
from .catalog import PaperCatalog
from .demo import run_demo
from .discovery import (
    ArxivProvider,
    DiscoveryQueryPlanner,
    FallbackAcademicProvider,
    OpenAlexProvider,
    SeedDiscoveryPipeline,
    SemanticScholarProvider,
)
from .llm import LLMConfig, LLMConfigurationError, OpenAICompatibleProvider
from .models import InnovationProfile, ResearchPlan, record_to_dict
from .extract import extract_catalog_paper
from .ingest import ingest_paper
from .project import ProjectConfigurationError, ResearchContext
from .toolkit import ResearchToolkit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunEvent:
    sequence: int
    event_type: str
    timestamp: str
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class SeedUpload:
    upload_id: str
    project_id: str
    filename: str
    path: Path
    size: int


@dataclass(frozen=True)
class SeedRunInput:
    topic: str
    innovation_points: tuple[str, ...]
    uploads: tuple[SeedUpload, ...]


@dataclass
class ManagedRun:
    run_id: str
    project_id: str
    workspace: Path
    mode: str
    objective: str | None
    simulate_failure: bool
    seed_input: SeedRunInput | None = None
    max_discovered_papers: int = 12
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    events: list[RunEvent] = field(default_factory=list)
    result: dict[str, Any] | None = None
    artifact_path: Path | None = None
    manifest_path: Path | None = None
    error: str | None = None
    finished: bool = False
    lock: Lock = field(default_factory=Lock, repr=False)

    def append(self, event_type: str, data: Mapping[str, Any]) -> RunEvent:
        with self.lock:
            event = RunEvent(
                sequence=len(self.events) + 1,
                event_type=event_type,
                timestamp=_now(),
                data=dict(data),
            )
            self.events.append(event)
            self.updated_at = event.timestamp
            if event_type == "provider_failed":
                self.status = "blocked"
            elif event_type == "recovery_started":
                self.status = "recovering"
            elif event_type == "run_completed":
                self.status = "completed"
            elif event_type == "run_stopped":
                self.status = str(data.get("status", "blocked"))
            elif event_type == "run_failed":
                self.status = "failed"
            elif self.status == "queued":
                self.status = "running"
            return event

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "run_id": self.run_id,
                "project_id": self.project_id,
                "mode": self.mode,
                "objective": self.objective,
                "status": self.status,
                "finished": self.finished,
                "simulate_failure": self.simulate_failure,
                "seed_discovery": self.seed_input is not None,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "event_count": len(self.events),
                "error": self.error,
                "result": self.result,
            }

    def events_after(self, sequence: int) -> tuple[RunEvent, ...]:
        with self.lock:
            return tuple(event for event in self.events if event.sequence > sequence)


class RunService:
    """Manage isolated background demo runs and their event streams."""

    def __init__(
        self,
        repository_root: str | Path,
        runtime_root: str | Path | None = None,
        *,
        step_delay: float = 0.08,
        llm_config_path: str | Path = "config/llm.json",
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.runtime_root = (
            Path(runtime_root).resolve()
            if runtime_root is not None
            else self.repository_root / ".runtime" / "web-runs"
        )
        self._runs: dict[str, ManagedRun] = {}
        self._uploads: dict[str, SeedUpload] = {}
        self._lock = Lock()
        self.step_delay = max(0, step_delay)
        config_path = Path(llm_config_path)
        self.llm_config_path = (
            config_path if config_path.is_absolute() else self.repository_root / config_path
        )

    def start(
        self,
        project_id: str,
        *,
        mode: str = "demo",
        objective: str | None = None,
        simulate_failure: bool = True,
        seed_input: SeedRunInput | None = None,
        max_discovered_papers: int = 12,
    ) -> ManagedRun:
        ResearchContext.load(self.repository_root, project_id)
        requested_objective = objective.strip() if objective else None
        if mode == "live":
            config = LLMConfig.load(self.llm_config_path)
            config.api_key()
        run_id = uuid4().hex[:12]
        run = ManagedRun(
            run_id=run_id,
            project_id=project_id,
            workspace=self.runtime_root / run_id,
            mode=mode,
            objective=requested_objective,
            simulate_failure=simulate_failure,
            seed_input=seed_input,
            max_discovered_papers=max_discovered_papers,
        )
        with self._lock:
            self._runs[run_id] = run
        run.append("run_started", {
            "run_id": run_id,
            "project_id": project_id,
            "mode": mode,
            "objective": requested_objective,
        })
        Thread(target=self._execute, args=(run,), daemon=True).start()
        return run

    def save_seed_pdf(
        self,
        project_id: str,
        filename: str,
        content: bytes,
    ) -> SeedUpload:
        ResearchContext.load(self.repository_root, project_id)
        clean_name = Path(filename).name.strip()
        if not clean_name or not clean_name.casefold().endswith(".pdf"):
            raise ValueError("Seed paper must have a .pdf filename")
        if not content or b"%PDF-" not in content[:1024]:
            raise ValueError("Uploaded seed paper is not a valid PDF")
        if len(content) > 25 * 1024 * 1024:
            raise ValueError("Each seed PDF must be 25 MB or smaller")
        upload_id = uuid4().hex
        directory = self.runtime_root / "_uploads" / upload_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / clean_name
        path.write_bytes(content)
        upload = SeedUpload(upload_id, project_id, clean_name, path, len(content))
        with self._lock:
            self._uploads[upload_id] = upload
        return upload

    def seed_uploads(
        self,
        project_id: str,
        upload_ids: tuple[str, ...],
    ) -> tuple[SeedUpload, ...]:
        with self._lock:
            uploads = tuple(self._uploads.get(upload_id) for upload_id in upload_ids)
        if any(upload is None for upload in uploads):
            raise ValueError("One or more seed uploads are unknown or expired")
        resolved = tuple(upload for upload in uploads if upload is not None)
        if any(upload.project_id != project_id for upload in resolved):
            raise ValueError("A seed upload belongs to another project")
        return resolved

    def get(self, run_id: str) -> ManagedRun:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def _execute(self, run: ManagedRun) -> None:
        try:
            if run.mode == "live":
                self._execute_live(run)
            else:
                self._execute_demo(run)
        except Exception as exc:
            with run.lock:
                run.error = f"{type(exc).__name__}: {exc}"
                run.updated_at = _now()
            run.append("run_failed", {"error": run.error})
            with run.lock:
                run.finished = True

    def _execute_demo(self, run: ManagedRun) -> None:
        completion_event: dict[str, Any] | None = None

        def emit(event_type: str, data: Mapping[str, Any]) -> None:
            nonlocal completion_event
            if event_type == "run_completed":
                completion_event = dict(data)
                return
            run.append(event_type, data)

        result = run_demo(
            self.repository_root,
            project_id=run.project_id,
            workspace=run.workspace,
            simulate_failure=run.simulate_failure,
            on_event=emit,
            step_delay=self.step_delay,
        )
        with run.lock:
            run.result = result.to_dict()
            run.artifact_path = result.artifact_path
            run.manifest_path = result.manifest_path
            run.updated_at = _now()
        run.append("run_completed", completion_event or {
            "status": result.resumed_status,
            "completed_task_ids": list(result.completed_task_ids),
            "artifact_path": str(result.artifact_path),
            "citation_guard_passed": True,
        })
        with run.lock:
            run.finished = True

    def _execute_live(self, run: ManagedRun) -> None:
        context = _clone_project_workspace(
            ResearchContext.load(self.repository_root, run.project_id),
            run.workspace,
            include_source_corpus=run.seed_input is None,
        )
        run.append("workspace_prepared", {
            "workspace": str(run.workspace),
            "project_id": run.project_id,
            "isolated": True,
        })
        config = LLMConfig.load(self.llm_config_path)
        discovery = None
        profile = None
        if run.seed_input is not None:
            context, seed_paper_ids = _ingest_seed_uploads(context, run.seed_input.uploads)
            profile = InnovationProfile(
                project_id=run.project_id,
                topic=run.seed_input.topic,
                innovation_points=run.seed_input.innovation_points,
                seed_paper_ids=seed_paper_ids,
            )
            context = _apply_innovation_profile(context, profile)
            model_provider = OpenAICompatibleProvider(config)
            discovery = SeedDiscoveryPipeline(
                context,
                FallbackAcademicProvider((
                    OpenAlexProvider(),
                    SemanticScholarProvider(),
                    ArxivProvider(),
                )),
                DiscoveryQueryPlanner(model_provider),
                on_event=run.append,
            ).run(
                profile,
                max_papers=run.max_discovered_papers,
            )
            if not ResearchToolkit(context).list_papers():
                raise ProjectConfigurationError(
                    "Seed discovery produced no readable papers"
                )
        objective = run.objective or (
            _related_work_objective(profile)
            if profile is not None
            else context.project.research_question
        )
        run.append("planner_started", {
            "objective": objective,
            "model": config.model,
        })
        agent = ResearchAgent.from_config(
            context,
            config,
            on_event=run.append,
        )
        prepared = agent.prepare_plan(objective)
        _emit_plan(run, prepared.plan)
        execution = agent.plan_executor.execute(prepared.plan)
        artifact_path, manifest_path = _find_primary_artifact(
            context,
            prepared.plan,
            agent,
        )
        result = {
            "mode": "live_model",
            "model": config.model,
            "plan_created": prepared.created,
            "plan": record_to_dict(prepared.plan),
            "execution": {
                "status": execution.status,
                "completed_task_ids": list(execution.completed_task_ids),
                "pending_task_ids": list(execution.pending_task_ids),
                "stopped_task_id": execution.stopped_task_id,
                "message": execution.message,
            },
            "output": {
                "artifact_path": str(artifact_path) if artifact_path else None,
                "manifest_path": str(manifest_path) if manifest_path else None,
            },
            "discovery": discovery.to_dict() if discovery is not None else None,
        }
        with run.lock:
            run.result = result
            run.artifact_path = artifact_path
            run.manifest_path = manifest_path
            run.updated_at = _now()
        if execution.status == "completed":
            run.append("run_completed", {
                "status": "completed",
                "completed_task_ids": list(execution.completed_task_ids),
                "artifact_path": str(artifact_path) if artifact_path else None,
                "citation_guard_passed": True,
            })
        else:
            run.append("run_stopped", {
                "status": execution.status,
                "task_id": execution.stopped_task_id,
                "message": execution.message,
            })
        with run.lock:
            run.finished = True


class StartRunRequest(BaseModel):
    project_id: str = Field(min_length=1)
    mode: Literal["demo", "live"] = "demo"
    objective: str | None = Field(default=None, max_length=2000)
    simulate_failure: bool = True
    topic: str | None = Field(default=None, max_length=500)
    innovation_points: list[str] = Field(default_factory=list, max_length=10)
    seed_upload_ids: list[str] = Field(default_factory=list, max_length=3)
    max_discovered_papers: int = Field(default=12, ge=3, le=20)


def create_app(
    repository_root: str | Path,
    *,
    runtime_root: str | Path | None = None,
    step_delay: float = 0.08,
    llm_config_path: str | Path = "config/llm.json",
) -> FastAPI:
    root = Path(repository_root).resolve()
    service = RunService(
        root,
        runtime_root,
        step_delay=step_delay,
        llm_config_path=llm_config_path,
    )
    static_root = Path(__file__).with_name("web_static")
    app = FastAPI(
        title="Evidence-Grounded Research Agent",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.run_service = service
    app.state.repository_root = root
    app.mount("/assets", StaticFiles(directory=static_root), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "research-agent",
            "live_model": _live_model_status(service.llm_config_path),
        }

    @app.get("/api/projects")
    async def projects() -> dict[str, Any]:
        return {"projects": _list_projects(root)}

    @app.get("/api/projects/{project_id}")
    async def project(project_id: str) -> dict[str, Any]:
        try:
            return _project_payload(ResearchContext.load(root, project_id))
        except ProjectConfigurationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/seed-papers", status_code=201)
    async def upload_seed_paper(project_id: str, request: Request) -> dict[str, Any]:
        filename = unquote(request.headers.get("x-filename", ""))
        try:
            upload = service.save_seed_pdf(project_id, filename, await request.body())
        except ProjectConfigurationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "upload_id": upload.upload_id,
            "filename": upload.filename,
            "size": upload.size,
        }

    @app.post("/api/runs", status_code=202)
    async def start_run(payload: StartRunRequest) -> dict[str, Any]:
        try:
            seed_input = _seed_input_from_request(payload, service)
            run = service.start(
                payload.project_id,
                mode=payload.mode,
                objective=payload.objective,
                simulate_failure=payload.simulate_failure,
                seed_input=seed_input,
                max_discovered_papers=payload.max_discovered_papers,
            )
        except ProjectConfigurationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LLMConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return run.snapshot()

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str) -> dict[str, Any]:
        return _run(service, run_id).snapshot()

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
        run = _run(service, run_id)
        header_id = request.headers.get("last-event-id")
        if header_id and header_id.isdigit():
            after = max(after, int(header_id))

        async def stream():
            sequence = after
            while True:
                emitted = False
                for event in run.events_after(sequence):
                    emitted = True
                    sequence = event.sequence
                    payload = json.dumps(event.to_dict(), ensure_ascii=False)
                    yield f"id: {event.sequence}\nevent: run_event\ndata: {payload}\n\n"
                snapshot = run.snapshot()
                if emitted:
                    continue
                if snapshot["finished"]:
                    break
                if await request.is_disconnected():
                    break
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/runs/{run_id}/artifact")
    async def run_artifact(run_id: str) -> JSONResponse:
        run = _run(service, run_id)
        snapshot = run.snapshot()
        if (
            snapshot["status"] != "completed"
            or run.artifact_path is None
            or run.manifest_path is None
        ):
            raise HTTPException(status_code=409, detail="Run artifact is not ready")
        try:
            content = run.artifact_path.read_text(encoding="utf-8")
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Cannot read run artifact") from exc
        return JSONResponse({
            "run_id": run_id,
            "content": content,
            "manifest": manifest,
        })

    return app


def _run(service: RunService, run_id: str) -> ManagedRun:
    try:
        return service.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc


def _live_model_status(config_path: Path) -> dict[str, Any]:
    try:
        config = LLMConfig.load(config_path)
        config.api_key()
    except LLMConfigurationError as exc:
        return {"ready": False, "reason": str(exc)}
    return {"ready": True, "model": config.model}


def _seed_input_from_request(
    payload: StartRunRequest,
    service: RunService,
) -> SeedRunInput | None:
    has_seed_input = bool(
        payload.topic or payload.innovation_points or payload.seed_upload_ids
    )
    if not has_seed_input:
        return None
    if payload.mode != "live":
        raise ValueError("Seed discovery is available only in live mode")
    topic = payload.topic.strip() if payload.topic else ""
    innovations = tuple(item.strip() for item in payload.innovation_points if item.strip())
    upload_ids = tuple(item.strip() for item in payload.seed_upload_ids if item.strip())
    if not topic:
        raise ValueError("Seed discovery requires a topic")
    if not innovations:
        raise ValueError("Seed discovery requires at least one innovation point")
    if not upload_ids:
        raise ValueError("Seed discovery requires 1 to 3 uploaded PDF papers")
    if len(set(upload_ids)) != len(upload_ids):
        raise ValueError("Seed PDF uploads must be unique")
    return SeedRunInput(
        topic=topic,
        innovation_points=innovations,
        uploads=service.seed_uploads(payload.project_id, upload_ids),
    )


def _clone_project_workspace(
    source: ResearchContext,
    workspace: Path,
    *,
    include_source_corpus: bool = True,
) -> ResearchContext:
    project_dir = workspace / "projects" / source.project_id
    project_dir.mkdir(parents=True, exist_ok=False)
    try:
        config = json.loads(source.config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectConfigurationError("Cannot clone project configuration") from exc
    config["uploaded_papers"] = []
    config["output_directory"] = f"projects/{source.project_id}/outputs"
    if not include_source_corpus:
        config["seed_papers"] = []
    (project_dir / "project.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_ids = None if include_source_corpus else set()
    for name in ("brief.md", "style.md", "state.yaml"):
        source_path = source.project_directory / name
        if source_path.is_file():
            shutil.copy2(source_path, project_dir / name)
    for name in ("catalog.jsonl", "bibliography.jsonl", "evidence.jsonl"):
        source_path = source.project_directory / name
        if source_path.is_file():
            _copy_filtered_jsonl(source_path, project_dir / name, selected_ids)
    for name in ("notes", "extracted"):
        source_path = source.project_directory / name
        if source_path.is_dir():
            target_path = project_dir / name
            target_path.mkdir()
            for item in source_path.iterdir():
                if not item.is_file():
                    continue
                if selected_ids is None or any(
                    item.name.startswith(paper_id.replace(":", "_"))
                    for paper_id in selected_ids
                ):
                    shutil.copy2(item, target_path / item.name)
    return ResearchContext.load(workspace, source.project_id)


def _ingest_seed_uploads(
    context: ResearchContext,
    uploads: tuple[SeedUpload, ...],
) -> tuple[ResearchContext, tuple[str, ...]]:
    papers_directory = context.project_directory / "papers"
    papers_directory.mkdir(parents=True, exist_ok=True)
    catalog = PaperCatalog(context.catalog_path)
    records = []
    for upload in uploads:
        target = papers_directory / f"{upload.upload_id[:12]}-{upload.filename}"
        shutil.copy2(upload.path, target)
        result = ingest_paper(
            target,
            catalog,
            title=_uploaded_pdf_title(target, upload.filename),
            source_type="uploaded_seed",
            tags=("seed",),
            path_base=context.repository_root,
        )
        extract_catalog_paper(
            catalog,
            result.record.paper_id,
            context.extraction_directory,
            path_base=context.repository_root,
        )
        records.append(result.record)
    config = json.loads(context.config_path.read_text(encoding="utf-8"))
    config["uploaded_papers"] = [str(record.file_path) for record in records]
    config["seed_papers"] = [record.title for record in records]
    context.config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return (
        ResearchContext.load(context.repository_root, context.project_id),
        tuple(record.paper_id for record in records),
    )


def _uploaded_pdf_title(path: Path, filename: str) -> str:
    fallback = Path(filename).stem.replace("_", " ").strip()
    try:
        metadata = PdfReader(path).metadata
        title = metadata.title if metadata is not None else None
    except Exception:
        return fallback
    return title.strip() if isinstance(title, str) and title.strip() else fallback


def _copy_filtered_jsonl(source: Path, target: Path, selected: set[str] | None) -> None:
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if selected is None:
            lines.append(line)
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProjectConfigurationError(f"Cannot clone invalid JSONL: {source}") from exc
        if value.get("paper_id") in selected:
            lines.append(line)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _apply_innovation_profile(
    context: ResearchContext,
    profile: InnovationProfile,
) -> ResearchContext:
    config = json.loads(context.config_path.read_text(encoding="utf-8"))
    innovations = "; ".join(profile.innovation_points)
    config["research_question"] = profile.topic
    config["description"] = (
        "Seed-grounded Related Work project. Proposed innovations: " + innovations
    )
    catalog = PaperCatalog(context.catalog_path)
    config["seed_papers"] = [
        paper.title
        for paper_id in profile.seed_paper_ids
        if (paper := catalog.get(paper_id)) is not None
    ]
    config["research_scope"] = [
        "Progressive method taxonomy and development trajectory",
        "Evidence-supported limitations of prior method families",
        "Careful mapping from prior limitations to the proposed innovations",
        "Separate direct paper evidence from cross-paper inference",
    ]
    context.config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ResearchContext.load(context.repository_root, context.project_id)


def _related_work_objective(profile: InnovationProfile) -> str:
    innovations = " | ".join(profile.innovation_points)
    return (
        f"Produce a concise evidence-grounded Related Work synthesis for: {profile.topic}. "
        "Organize the literature into a progressive taxonomy of method families, "
        "explain how the approaches developed, and synthesize shared limitations "
        "instead of listing papers one by one. Map supported limitations to these "
        f"proposed innovations: {innovations}. Do not manufacture a research gap; "
        "label cross-paper conclusions as inference and retain citation provenance."
    )


def _emit_plan(run: ManagedRun, plan: ResearchPlan) -> None:
    run.append("plan_validated", {
        "plan_id": plan.plan_id,
        "objective": plan.objective,
        "tasks": [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question": task.question,
                "dependencies": list(task.dependencies),
            }
            for task in plan.tasks
        ],
    })


def _find_primary_artifact(
    context: ResearchContext,
    plan: ResearchPlan,
    agent: ResearchAgent,
) -> tuple[Path | None, Path | None]:
    toolkit = agent.plan_executor.task_executor.dispatcher.toolkit
    for task in reversed(plan.tasks):
        state = agent.plan_executor.task_executor.state_store.load(task.task_id)
        if state is None:
            continue
        for artifact_id in reversed(state.artifact_ids):
            output = toolkit.read_output(artifact_id)
            path = output.artifact.path
            if not path.is_absolute():
                path = context.repository_root / path
            return path, path.with_name(f"{path.name}.manifest.json")
    return None, None


def _list_projects(root: Path) -> list[dict[str, Any]]:
    projects_root = root / "projects"
    values: list[dict[str, Any]] = []
    if not projects_root.is_dir():
        return values
    for directory in sorted(projects_root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        try:
            values.append(_project_payload(ResearchContext.load(root, directory.name)))
        except ProjectConfigurationError:
            continue
    return values


def _project_payload(context: ResearchContext) -> dict[str, Any]:
    tools = ResearchToolkit(context)
    papers = tools.list_papers()
    evidence = tools.list_evidence()
    note_count = sum(tools.load_note(paper.paper_id) is not None for paper in papers)
    return {
        "project_id": context.project_id,
        "title": context.project.title,
        "research_question": context.project.research_question,
        "description": context.project.description,
        "research_scope": list(context.project.research_scope),
        "seed_papers": list(context.project.seed_papers),
        "stats": {
            "papers": len(papers),
            "notes": note_count,
            "evidence": len(evidence),
        },
        "papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "tags": list(paper.tags),
                "has_note": tools.load_note(paper.paper_id) is not None,
                "evidence_count": sum(item.paper_id == paper.paper_id for item in evidence),
            }
            for paper in papers
        ],
    }

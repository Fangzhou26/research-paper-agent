import asyncio
from io import BytesIO
import json
from pathlib import Path

import httpx
from pypdf import PdfWriter

from research_agent.agent import ResearchAgent
from research_agent.models import ResearchPlan, ResearchTask
from research_agent.plan_execution import PlanExecutionResult
from research_agent.planning import PlanWriteResult
from research_agent.web import create_app


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_web_run_reaches_audited_artifact(tmp_path: Path) -> None:
    app = create_app(
        REPOSITORY_ROOT,
        runtime_root=tmp_path / "runs",
        step_delay=0,
    )

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            assert (await client.get("/api/health")).json()["status"] == "ok"
            project = await client.get("/api/projects/demo")
            assert project.status_code == 200
            assert project.json()["stats"]["papers"] >= 1

            started = await client.post(
                "/api/runs",
                json={
                    "project_id": "demo",
                    "simulate_failure": True,
                },
            )
            assert started.status_code == 202
            run_id = started.json()["run_id"]
            run = await _wait_for_run(client, run_id)

            assert run["status"] == "completed"
            assert run["result"]["failure"]["first_attempt_status"] == "blocked"
            assert run["result"]["citation_guard"]["unknown_keys"] == ["invented2026"]

            events = await client.get(f"/api/runs/{run_id}/events")
            assert events.status_code == 200
            assert "event: run_event" in events.text
            assert '"event_type": "provider_failed"' in events.text
            assert '"event_type": "recovery_started"' in events.text
            assert '"event_type": "run_completed"' in events.text

            artifact = await client.get(f"/api/runs/{run_id}/artifact")
            assert artifact.status_code == 200
            payload = artifact.json()
            assert "Evidence-backed synthesis" in payload["content"]
            assert payload["manifest"]["paper_ids"]
            assert payload["manifest"]["evidence_ids"]

    asyncio.run(scenario())


def test_web_rejects_unknown_project(tmp_path: Path) -> None:
    app = create_app(REPOSITORY_ROOT, runtime_root=tmp_path / "runs", step_delay=0)

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/runs",
                json={"project_id": "missing-project", "simulate_failure": False},
            )

    response = asyncio.run(scenario())
    assert response.status_code == 404


def test_live_run_requires_configured_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.json"
    config_path.write_text(json.dumps({
        "provider": "openai_compatible",
        "model": "test-model",
        "api_key_env": "RESEARCH_AGENT_TEST_KEY_NOT_SET",
    }), encoding="utf-8")
    app = create_app(
        REPOSITORY_ROOT,
        runtime_root=tmp_path / "runs",
        step_delay=0,
        llm_config_path=config_path,
    )

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/api/health")
            started = await client.post(
                "/api/runs",
                json={
                    "project_id": "demo",
                    "mode": "live",
                    "objective": "Compare evidence in the configured corpus.",
                },
            )
            return health, started

    health, started = asyncio.run(scenario())
    assert health.json()["live_model"]["ready"] is False
    assert started.status_code == 503
    assert "RESEARCH_AGENT_TEST_KEY_NOT_SET" in started.json()["detail"]


def test_seed_mode_validates_required_inputs(tmp_path: Path) -> None:
    app = create_app(REPOSITORY_ROOT, runtime_root=tmp_path / "runs", step_delay=0)

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/runs", json={
                "project_id": "demo",
                "mode": "live",
                "topic": "Perceptive humanoid tracking",
                "seed_upload_ids": ["not-used-before-innovation-validation"],
                "innovation_points": [],
            })

    response = asyncio.run(scenario())
    assert response.status_code == 422
    assert "innovation point" in response.json()["detail"]


def test_seed_upload_rejects_non_pdf(tmp_path: Path) -> None:
    app = create_app(REPOSITORY_ROOT, runtime_root=tmp_path / "runs", step_delay=0)

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/api/projects/demo/seed-papers",
                headers={"X-Filename": "notes.txt", "Content-Type": "text/plain"},
                content=b"not a pdf",
            )

    response = asyncio.run(scenario())
    assert response.status_code == 422
    assert ".pdf" in response.json()["detail"]


def test_live_run_uses_isolated_project_and_streams_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "llm.json"
    config_path.write_text(json.dumps({
        "provider": "openai_compatible",
        "model": "test-model",
        "api_key_env": "RESEARCH_AGENT_TEST_KEY",
    }), encoding="utf-8")
    monkeypatch.setenv("RESEARCH_AGENT_TEST_KEY", "test-secret")
    class FakeDiscoveryResult:
        def to_dict(self):
            return {
                "accepted_paper_ids": ["local-seed"],
                "warnings": [],
            }

    monkeypatch.setattr(
        "research_agent.web.SeedDiscoveryPipeline.run",
        lambda self, profile, max_papers: FakeDiscoveryResult(),
    )

    class FakeAgent:
        def __init__(self, context, on_event) -> None:
            self.context = context
            self.on_event = on_event
            self.plan = ResearchPlan(
                plan_id="live-test-plan",
                project_id=context.project_id,
                objective="Inspect the isolated corpus.",
                tasks=(ResearchTask(
                    task_id="live-overview",
                    question="What evidence is available?",
                    task_type="corpus_overview",
                ),),
                deliverables=("Evidence overview",),
                completion_criteria=("The corpus is inspected",),
            )
            state_store = type("StateStore", (), {"load": lambda self, task_id: None})()
            task_executor = type("TaskExecutor", (), {
                "state_store": state_store,
                "dispatcher": type("Dispatcher", (), {"toolkit": None})(),
            })()
            self.plan_executor = type("PlanExecutor", (), {
                "task_executor": task_executor,
                "execute": self.execute,
            })()

        def prepare_plan(self, objective):
            return PlanWriteResult(self.plan, created=True, updated=False)

        def execute(self, plan):
            self.on_event("task_started", {
                "task_id": "live-overview",
                "task_type": "corpus_overview",
                "question": "What evidence is available?",
            })
            self.on_event("task_finishing", {
                "task_id": "live-overview",
                "status": "completed",
                "summary": "Corpus inspected.",
            })
            return PlanExecutionResult(
                status="completed",
                completed_task_ids=("live-overview",),
                pending_task_ids=(),
                task_results=(),
                message="All tasks completed.",
            )

    monkeypatch.setattr(
        ResearchAgent,
        "from_config",
        classmethod(lambda cls, context, config, **kwargs: FakeAgent(
            context,
            kwargs["on_event"],
        )),
    )
    runtime_root = tmp_path / "runs"
    app = create_app(
        REPOSITORY_ROOT,
        runtime_root=runtime_root,
        step_delay=0,
        llm_config_path=config_path,
    )

    async def scenario() -> tuple[dict, str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            uploaded = await client.post(
                "/api/projects/demo/seed-papers",
                headers={"X-Filename": "Seed_Tracking_Paper.pdf", "Content-Type": "application/pdf"},
                content=_pdf_bytes(),
            )
            assert uploaded.status_code == 201
            started = await client.post("/api/runs", json={
                "project_id": "demo",
                "mode": "live",
                "topic": "Perceptive humanoid motion tracking",
                "innovation_points": ["Fuse motion intent with terrain observations"],
                "seed_upload_ids": [uploaded.json()["upload_id"]],
                "max_discovered_papers": 6,
            })
            assert started.status_code == 202
            run_id = started.json()["run_id"]
            run = await _wait_for_run(client, run_id)
            events = (await client.get(f"/api/runs/{run_id}/events")).text
            return run, events

    run, events = asyncio.run(scenario())
    assert run["status"] == "completed"
    assert run["mode"] == "live"
    assert run["seed_discovery"] is True
    assert run["result"]["model"] == "test-model"
    assert run["result"]["discovery"]["accepted_paper_ids"] == ["local-seed"]
    assert '"event_type": "planner_started"' in events
    assert '"event_type": "plan_validated"' in events
    assert '"event_type": "task_started"' in events
    cloned = runtime_root / run["run_id"] / "projects" / "demo"
    assert (cloned / "catalog.jsonl").is_file()
    assert len(list((cloned / "papers").glob("*.pdf"))) == 1
    assert len(list((cloned / "extracted").glob("*.pages.jsonl"))) == 1
    seed_record = json.loads((cloned / "catalog.jsonl").read_text().splitlines()[0])
    assert seed_record["title"] == "Seed Tracking Paper"
    assert not (cloned / "research_plan.json").exists()


def _pdf_bytes() -> bytes:
    stream = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": "Seed Tracking Paper"})
    writer.write(stream)
    return stream.getvalue()


async def _wait_for_run(client: httpx.AsyncClient, run_id: str) -> dict:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        payload = (await client.get(f"/api/runs/{run_id}")).json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        await asyncio.sleep(0.01)
    raise AssertionError("web run did not reach a terminal state")

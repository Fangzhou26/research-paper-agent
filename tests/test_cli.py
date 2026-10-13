import json
from pathlib import Path

from research_agent import ResearchContext, ResearchPlan, ResearchPlanStore, ResearchTask, RunState, RunStateStore
from research_agent.cli import main


def repository(tmp_path: Path) -> Path:
    project_dir = tmp_path / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (tmp_path / "projects" / "active_project").write_text("demo\n", encoding="utf-8")
    (project_dir / "project.json").write_text(json.dumps({
        "title": "Demo project",
        "research_question": "What does the corpus show?",
        "description": None,
        "uploaded_papers": [],
        "seed_papers": [],
        "research_scope": [],
        "output_directory": "projects/demo/outputs",
    }), encoding="utf-8")
    return tmp_path


def plan() -> ResearchPlan:
    return ResearchPlan(
        plan_id="demo-plan",
        project_id="demo",
        objective="What does the corpus show?",
        subquestions=("What is available?",),
        tasks=(ResearchTask(
            task_id="overview",
            question="What is available?",
            task_type="corpus_overview",
        ),),
        deliverables=("Overview",),
        completion_criteria=("Corpus inspected",),
    )


def test_status_reports_project_without_plan_offline(tmp_path: Path, capsys) -> None:
    root = repository(tmp_path)

    exit_code = main(["--root", str(root), "status"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "no_plan"
    assert payload["project_id"] == "demo"


def test_status_reports_persisted_task_checkpoint(tmp_path: Path, capsys) -> None:
    root = repository(tmp_path)
    context = ResearchContext.load_active(root)
    ResearchPlanStore(context).save(plan())
    RunStateStore(context).save(RunState(
        task_id="overview",
        project_id="demo",
        task_type="corpus_overview",
        status="needs_input",
        current_stage="finished",
        open_questions=("Which sources should be included?",),
        summary="Scope needs confirmation.",
    ))

    exit_code = main(["--root", str(root), "--project", "demo", "status"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "needs_input"
    assert payload["tasks"][0]["stage"] == "finished"
    assert payload["tasks"][0]["open_questions"] == ["Which sources should be included?"]


def test_status_reports_stored_plan_as_planned_before_execution(tmp_path: Path, capsys) -> None:
    root = repository(tmp_path)
    context = ResearchContext.load_active(root)
    ResearchPlanStore(context).save(plan())

    exit_code = main(["--root", str(root), "status"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "planned"


def test_cli_rejects_non_positive_round_limit() -> None:
    try:
        main(["run", "--max-task-rounds", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should reject a non-positive limit")


def test_demo_runs_without_llm_config_or_api_key(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "demo"

    exit_code = main([
        "--root", str(Path(__file__).resolve().parents[1]),
        "--project", "demo",
        "--llm-config", "missing.json",
        "demo",
        "--workspace", str(workspace),
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure"]["first_attempt_status"] == "blocked"
    assert payload["recovery"]["status"] == "completed"
    assert payload["citation_guard"]["unknown_keys"] == ["invented2026"]

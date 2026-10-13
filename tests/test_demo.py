from pathlib import Path

from research_agent.demo import run_demo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_offline_demo_recovers_and_writes_audited_artifact(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []
    result = run_demo(
        REPOSITORY_ROOT,
        project_id="demo",
        workspace=tmp_path / "demo",
        on_event=lambda event_type, data: events.append((event_type, dict(data))),
    )

    assert result.first_attempt_status == "blocked"
    assert result.plan_id == "interview-demo-plan"
    assert result.planned_task_ids == (
        "demo-analyze",
        "demo-synthesize",
        "demo-audit",
    )
    assert result.first_attempt_stopped_task == "demo-analyze"
    assert result.resumed_status == "completed"
    assert result.completed_task_ids == (
        "demo-analyze",
        "demo-synthesize",
        "demo-audit",
    )
    assert result.unsafe_citation_keys == ("invented2026",)
    assert result.artifact_path.is_file()
    assert result.manifest_path.is_file()
    assert "search_pages" in result.tool_trace
    assert "write_output" in result.tool_trace
    assert result.tool_trace.count("audit_citations") == 2
    event_types = [event_type for event_type, _ in events]
    assert event_types.index("plan_validated") < event_types.index("provider_failed")
    assert event_types.index("provider_failed") < event_types.index("recovery_started")
    assert event_types.index("citation_rejected") < event_types.index("run_completed")


def test_offline_demo_can_run_without_fault_injection(tmp_path: Path) -> None:
    result = run_demo(
        REPOSITORY_ROOT,
        project_id="demo",
        workspace=tmp_path / "demo",
        simulate_failure=False,
    )

    assert result.first_attempt_status == "completed"
    assert result.resumed_status == "completed"
    assert result.to_dict()["failure"]["injected"] is False
    assert result.to_dict()["recovery"]["action"] == "not_required"

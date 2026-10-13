"""Thin command-line interface for the research-agent MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .agent import ResearchAgent, ResearchAgentError
from .demo import DemoError, run_demo
from .execution import RunStateStore
from .llm import LLMConfig, LLMConfigurationError, LLMProviderError
from .llm_planner import PlannerError
from .models import record_to_dict
from .planning import PlanError, ResearchPlanStore
from .project import ProjectConfigurationError, ResearchContext


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        root = Path(args.root).resolve()
        context = (
            ResearchContext.load(root, args.project)
            if args.project
            else ResearchContext.load_active(root)
        )
        if args.command == "status":
            _print_json(_status_payload(context))
            return 0
        if args.command == "demo":
            result = run_demo(
                root,
                project_id=context.project_id,
                workspace=args.workspace,
                simulate_failure=not args.no_failure,
            )
            _print_json(result.to_dict())
            return 0
        if args.command == "serve":
            import uvicorn

            from .web import create_app

            uvicorn.run(
                create_app(root),
                host=args.host,
                port=args.port,
                log_level="info",
            )
            return 0

        config_path = Path(args.llm_config)
        if not config_path.is_absolute():
            config_path = root / config_path
        agent = ResearchAgent.from_config(
            context,
            LLMConfig.load(config_path),
            max_task_rounds=args.max_task_rounds,
        )
        if args.command == "plan":
            prepared = (
                agent.replace_unstarted_plan(args.objective)
                if args.replace_unstarted
                else agent.prepare_plan(args.objective)
            )
            _print_json({
                "status": "planned",
                "created": prepared.created,
                "path": str(agent.plan_store.path),
                "plan": record_to_dict(prepared.plan),
            })
            return 0
        result = agent.run(args.objective, retry_blocked=args.retry_blocked)
        _print_json({
            "status": result.execution.status,
            "plan_created": result.plan_created,
            "plan_id": result.plan.plan_id,
            "completed_task_ids": list(result.execution.completed_task_ids),
            "pending_task_ids": list(result.execution.pending_task_ids),
            "stopped_task_id": result.execution.stopped_task_id,
            "message": result.execution.message,
        })
        return {"completed": 0, "needs_input": 2, "blocked": 3}.get(
            result.execution.status, 1
        )
    except (
        LLMConfigurationError,
        LLMProviderError,
        DemoError,
        PlanError,
        PlannerError,
        ProjectConfigurationError,
        ResearchAgentError,
    ) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent")
    parser.add_argument("--root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--project", help="Project id (default: projects/active_project)")
    parser.add_argument("--llm-config", default="config/llm.json", help="LLM config path relative to root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create or display the stored research plan")
    plan.add_argument("--objective", help="Research objective (default: project research question)")
    plan.add_argument(
        "--replace-unstarted",
        action="store_true",
        help="Replace the stored plan only if no task checkpoint exists",
    )
    plan.add_argument("--max-task-rounds", type=_positive_int, default=12, help=argparse.SUPPRESS)

    run = subparsers.add_parser("run", help="Create if needed, then execute or resume the research plan")
    run.add_argument("--objective", help="Must match the stored objective when a plan exists")
    run.add_argument("--max-task-rounds", type=_positive_int, default=12)
    run.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Explicitly retry the first blocked task from its checkpoint",
    )

    demo = subparsers.add_parser(
        "demo",
        help="Run a deterministic offline end-to-end demo without an API key",
    )
    demo.add_argument(
        "--workspace",
        help="Empty directory for the isolated demo (default: a new directory under /tmp)",
    )
    demo.add_argument(
        "--no-failure",
        action="store_true",
        help="Skip the simulated provider failure and recovery step",
    )

    serve = subparsers.add_parser(
        "serve",
        help="Start the local research-agent web workspace",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8000)

    subparsers.add_parser("status", help="Show plan and task checkpoints without calling the LLM")
    return parser


def _status_payload(context: ResearchContext) -> dict[str, Any]:
    plan = ResearchPlanStore(context).load()
    if plan is None:
        return {
            "status": "no_plan",
            "project_id": context.project_id,
            "plan_path": str(context.research_plan_path),
            "tasks": [],
        }
    state_store = RunStateStore(context)
    tasks = []
    has_checkpoint = False
    for task in plan.tasks:
        state = state_store.load(task.task_id)
        has_checkpoint = has_checkpoint or state is not None
        tasks.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dependencies": list(task.dependencies),
            "status": state.status if state is not None else task.status,
            "stage": state.current_stage if state is not None else None,
            "summary": state.summary if state is not None else None,
            "open_questions": list(state.open_questions) if state is not None else [],
        })
    statuses = {task["status"] for task in tasks}
    overall = (
        "needs_input" if "needs_input" in statuses
        else "blocked" if "blocked" in statuses
        else "completed" if statuses == {"completed"}
        else "planned" if not has_checkpoint
        else "in_progress"
    )
    return {
        "status": overall,
        "project_id": context.project_id,
        "plan_id": plan.plan_id,
        "objective": plan.objective,
        "plan_path": str(context.research_plan_path),
        "tasks": tasks,
    }


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

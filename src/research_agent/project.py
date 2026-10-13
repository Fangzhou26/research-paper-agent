"""Project and repository context models.

Project-specific content belongs in a project JSON file. This module only
defines the generic schema, path resolution, and validation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


ACTIVE_PROJECT_FILE = Path("projects/active_project")
PROJECT_CONFIG_NAME = "project.json"


class ProjectConfigurationError(ValueError):
    """Raised when a research project configuration is missing or invalid."""


@dataclass(frozen=True)
class ResearchProject:
    """Topic-specific inputs and destinations for one research project."""

    title: str
    research_question: str
    description: str | None
    uploaded_papers: tuple[Path, ...]
    seed_papers: tuple[str, ...]
    research_scope: tuple[str, ...]
    output_directory: Path


@dataclass(frozen=True)
class ResearchContext:
    """A research project resolved within a repository workspace."""

    repository_root: Path
    project_id: str
    config_path: Path
    project: ResearchProject

    @property
    def project_directory(self) -> Path:
        return self.config_path.parent

    @property
    def brief_path(self) -> Path:
        return self.project_directory / "brief.md"

    @property
    def style_path(self) -> Path:
        return self.project_directory / "style.md"

    @property
    def state_path(self) -> Path:
        return self.project_directory / "state.yaml"

    @property
    def catalog_path(self) -> Path:
        return self.project_directory / "catalog.jsonl"

    @property
    def extraction_directory(self) -> Path:
        return self.project_directory / "extracted"

    @property
    def notes_directory(self) -> Path:
        return self.project_directory / "notes"

    @property
    def evidence_path(self) -> Path:
        return self.project_directory / "evidence.jsonl"

    @property
    def bibliography_path(self) -> Path:
        return self.project_directory / "bibliography.jsonl"

    @property
    def artifacts_path(self) -> Path:
        return self.project_directory / "artifacts.jsonl"

    @property
    def research_plan_path(self) -> Path:
        return self.project_directory / "research_plan.json"

    @property
    def runs_directory(self) -> Path:
        return self.project_directory / "runs"

    @classmethod
    def load_active(cls, repository_root: str | Path) -> "ResearchContext":
        """Load the project selected by `projects/active_project`."""

        root = Path(repository_root).resolve()
        selector = root / ACTIVE_PROJECT_FILE
        try:
            project_id = selector.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ProjectConfigurationError(
                f"Cannot read active project selector: {selector}"
            ) from exc
        if not project_id:
            raise ProjectConfigurationError(f"Active project selector is empty: {selector}")
        return cls.load(root, project_id)

    @classmethod
    def load(cls, repository_root: str | Path, project_id: str) -> "ResearchContext":
        """Load a named project and resolve its configured paths."""

        root = Path(repository_root).resolve()
        _validate_project_id(project_id)
        config_path = root / "projects" / project_id / PROJECT_CONFIG_NAME
        data = _read_json_object(config_path)
        project = _project_from_mapping(data, root)
        return cls(
            repository_root=root,
            project_id=project_id,
            config_path=config_path,
            project=project,
        )

    def missing_uploaded_papers(self) -> tuple[Path, ...]:
        """Return configured uploaded-paper paths that are not present."""

        return tuple(path for path in self.project.uploaded_papers if not path.is_file())


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProjectConfigurationError(f"Cannot read project config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectConfigurationError(f"Invalid JSON in project config: {path}") from exc
    if not isinstance(value, dict):
        raise ProjectConfigurationError(f"Project config must contain a JSON object: {path}")
    return value


def _project_from_mapping(data: Mapping[str, Any], root: Path) -> ResearchProject:
    title = _required_text(data, "title")
    research_question = _required_text(data, "research_question")
    description = _optional_text(data, "description")
    uploaded = tuple(
        _resolve_workspace_path(root, path)
        for path in _string_list(data, "uploaded_papers")
    )
    seed_papers = tuple(_string_list(data, "seed_papers"))
    research_scope = tuple(_string_list(data, "research_scope"))
    output_directory = _resolve_workspace_path(
        root, _required_text(data, "output_directory")
    )
    return ResearchProject(
        title=title,
        research_question=research_question,
        description=description,
        uploaded_papers=uploaded,
        seed_papers=seed_papers,
        research_scope=research_scope,
        output_directory=output_directory,
    )


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProjectConfigurationError(f"'{key}' must be a non-empty string")
    return value.strip()


def _optional_text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProjectConfigurationError(f"'{key}' must be a string or null")
    return value.strip() or None


def _string_list(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ProjectConfigurationError(f"'{key}' must be a list of non-empty strings")
    return [item.strip() for item in value]


def _resolve_workspace_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_project_id(project_id: str) -> None:
    if not project_id or Path(project_id).name != project_id or project_id in {".", ".."}:
        raise ProjectConfigurationError(f"Invalid project id: {project_id!r}")

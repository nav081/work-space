"""Resolve project roots and enforce project-relative filesystem access."""

from __future__ import annotations

from pathlib import Path


def resolve_project_path(value: str) -> Path:
    """Resolve a valid project directory selected by the user."""
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("The selected project path is not a directory.")
    return path


def safe_project_file(root: Path, relative_path: str) -> Path:
    """Resolve a file path while preventing access outside the project root."""
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("Use a project-relative file path.")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("File access must stay inside the selected project.") from error
    return candidate

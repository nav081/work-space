"""Project-scoped file read, search, and write tool adapters."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import StructuredTool

from threadline_backend.core.constants import (
    IGNORED_SEARCH_DIRECTORIES,
    PROJECT_FILE_READ_LIMIT_BYTES,
    PROJECT_FILE_READ_LIMIT_LABEL,
    PROJECT_FILE_WRITE_LIMIT_BYTES,
    PROJECT_FILE_WRITE_LIMIT_LABEL,
    PROJECT_SEARCH_LINE_LIMIT,
    PROJECT_SEARCH_MATCH_LIMIT,
)
from threadline_backend.infrastructure.project_paths import safe_project_file


def _read_file(root: Path, relative_path: str) -> str:
    """Read a bounded UTF-8 file within the selected project."""
    target = safe_project_file(root, relative_path)
    if not target.is_file():
        return "File not found."
    if target.stat().st_size > PROJECT_FILE_READ_LIMIT_BYTES:
        return f"File is too large to read (limit: {PROJECT_FILE_READ_LIMIT_LABEL})."
    return target.read_text(encoding="utf-8", errors="replace")


def _search_codebase(root: Path, query: str) -> str:
    """Search project text files while skipping generated and hidden folders."""
    matches: list[str] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in IGNORED_SEARCH_DIRECTORIES and not name.startswith(".")]
        for filename in files:
            path = Path(current) / filename
            try:
                path.relative_to(root)
                if path.stat().st_size > PROJECT_FILE_READ_LIMIT_BYTES:
                    continue
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if query.casefold() in line.casefold():
                        matches.append(f"{path.relative_to(root)}:{line_number}: {line[:PROJECT_SEARCH_LINE_LIMIT]}")
                        if len(matches) == PROJECT_SEARCH_MATCH_LIMIT:
                            return "\n".join(matches)
            except (OSError, UnicodeError):
                continue
    return "\n".join(matches) if matches else "No matches found."


def _write_file(root: Path, relative_path: str, content: str) -> str:
    """Write a bounded UTF-8 file inside the selected project."""
    if len(content.encode("utf-8")) > PROJECT_FILE_WRITE_LIMIT_BYTES:
        return f"Write rejected: content exceeds the {PROJECT_FILE_WRITE_LIMIT_LABEL} limit."
    target = safe_project_file(root, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {target.relative_to(root)} ({len(content)} characters)."


def build_file_tools(root: Path, enabled: set[str]) -> list[StructuredTool]:
    """Build only the project file tools explicitly enabled for this agent."""
    tools: list[StructuredTool] = []
    if "Read files" in enabled:
        def read_file(relative_path: str) -> str:
            """Read a selected project file."""
            return _read_file(root, relative_path)

        tools.append(StructuredTool.from_function(read_file, name="read_project_file"))
    if "Search codebase" in enabled:
        def search_codebase(query: str) -> str:
            """Find matching text in the selected project."""
            return _search_codebase(root, query)

        tools.append(StructuredTool.from_function(search_codebase, name="search_project"))
    if "Edit files" in enabled:
        def write_file(relative_path: str, content: str) -> str:
            """Write a selected project file."""
            return _write_file(root, relative_path, content)

        tools.append(StructuredTool.from_function(write_file, name="write_project_file"))
    return tools

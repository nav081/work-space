"""Assemble project-scoped tool adapters from an agent's granted permissions."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from threadline_backend.infrastructure.command_tools import build_command_tools
from threadline_backend.infrastructure.dependency_tools import build_dependency_tools
from threadline_backend.infrastructure.file_tools import build_file_tools


def build_project_tools(root: Path, enabled: set[str]) -> list[StructuredTool]:
    """Create only the tools enabled for one agent."""
    return [
        *build_file_tools(root, enabled),
        *build_command_tools(root, enabled),
        *build_dependency_tools(root, enabled),
    ]

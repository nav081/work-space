"""Validated request and state contracts shared by API and workflow layers."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from threadline_backend.core.constants import (
    AGENT_DETAIL_MAX_LENGTH,
    AGENT_ID_MAX_LENGTH,
    AGENT_INSTRUCTION_MAX_LENGTH,
    AGENT_KIND_MAX_LENGTH,
    AGENT_NAME_MAX_LENGTH,
    AGENT_TOOL_LIMIT,
    DEFAULT_AGENT_MAX_RETRIES,
    EDGE_ID_MAX_LENGTH,
    EDGE_LABEL_MAX_LENGTH,
    MAX_AGENT_RETRIES,
    MODEL_ID_MAX_LENGTH,
    OUTCOME_EDGE_ID_MAX_LENGTH,
    PROJECT_PATH_MAX_LENGTH,
    REQUIREMENT_MAX_LENGTH,
    WORKFLOW_AGENT_LIMIT,
    WORKFLOW_EDGE_LIMIT,
)


class AgentConfig(BaseModel):
    """Agent settings accepted by a saved workflow run."""

    id: str = Field(min_length=1, max_length=AGENT_ID_MAX_LENGTH, pattern=r"^[A-Za-z0-9_-]+$")
    kind: str = Field(min_length=1, max_length=AGENT_KIND_MAX_LENGTH)
    name: str = Field(min_length=1, max_length=AGENT_NAME_MAX_LENGTH)
    detail: str = Field(default="", max_length=AGENT_DETAIL_MAX_LENGTH)
    instruction: str = Field(min_length=1, max_length=AGENT_INSTRUCTION_MAX_LENGTH)
    model: str = Field(min_length=1, max_length=MODEL_ID_MAX_LENGTH)
    tools: list[str] = Field(default_factory=list, max_length=AGENT_TOOL_LIMIT)
    success_edge_id: str | None = Field(default=None, max_length=OUTCOME_EDGE_ID_MAX_LENGTH)
    failure_edge_id: str | None = Field(default=None, max_length=OUTCOME_EDGE_ID_MAX_LENGTH)
    max_retries: int = Field(default=DEFAULT_AGENT_MAX_RETRIES, ge=0, le=MAX_AGENT_RETRIES)


class EdgeConfig(BaseModel):
    """Directed agent connection and its handoff message."""

    id: str = Field(default="", max_length=EDGE_ID_MAX_LENGTH)
    source: str = Field(min_length=1, max_length=AGENT_ID_MAX_LENGTH)
    target: str = Field(min_length=1, max_length=AGENT_ID_MAX_LENGTH)
    label: str = Field(default="", max_length=EDGE_LABEL_MAX_LENGTH)


class RunRequest(BaseModel):
    """Validated workflow execution request."""

    project_path: str = Field(min_length=1, max_length=PROJECT_PATH_MAX_LENGTH)
    requirement: str = Field(min_length=1, max_length=REQUIREMENT_MAX_LENGTH)
    agents: list[AgentConfig] = Field(min_length=1, max_length=WORKFLOW_AGENT_LIMIT)
    edges: list[EdgeConfig] = Field(default_factory=list, max_length=WORKFLOW_EDGE_LIMIT)


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge per-agent state updates, preferring the newest value."""
    return {**left, **right}


def _merge_handoffs(
    left: dict[str, list[dict[str, str]]],
    right: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    """Append routed handoff context independently for each destination."""
    merged = {target: list(contexts) for target, contexts in left.items()}
    for target, contexts in right.items():
        merged.setdefault(target, []).extend(contexts)
    return merged


def _merge_test_commands(
    left: dict[str, list[list[str]]],
    right: dict[str, list[list[str]]],
) -> dict[str, list[list[str]]]:
    """Accumulate observed test commands for each Tester agent."""
    merged = {agent_id: list(commands) for agent_id, commands in left.items()}
    for agent_id, commands in right.items():
        merged.setdefault(agent_id, []).extend(commands)
    return merged


class WorkflowState(TypedDict):
    """State carried between agents by the workflow graph."""

    requirement: str
    project_path: str
    messages: Annotated[list[dict[str, str]], lambda left, right: left + right]
    decisions: Annotated[dict[str, str], _merge_dicts]
    review_counts: Annotated[dict[str, int], _merge_dicts]
    handoffs: Annotated[dict[str, list[dict[str, str]]], _merge_handoffs]
    test_commands: Annotated[dict[str, list[list[str]]], _merge_test_commands]

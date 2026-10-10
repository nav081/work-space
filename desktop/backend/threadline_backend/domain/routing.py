"""Outcome classification and directed workflow routing policies."""

from __future__ import annotations

from threadline_backend.core.constants import (
    AGENT_FAILURE_MARKER,
    AGENT_SUCCESS_MARKER,
    DEVELOPER_KIND,
    END_NODE_ID,
    REVIEW_APPROVAL_LABELS,
    REVIEW_CHANGE_LABELS,
    REVIEWER_KINDS,
    TESTER_KINDS,
    TEST_FAILURE_LABELS,
    TEST_SUCCESS_LABELS,
)
from threadline_backend.domain.contracts import AgentConfig, EdgeConfig, WorkflowState


def agent_result_decision(content: str, command_failed: bool) -> str:
    """Classify a configured agent result, letting command errors take precedence."""
    normalized = content.casefold()
    if command_failed or AGENT_FAILURE_MARKER in normalized:
        return "fail"
    return "pass" if AGENT_SUCCESS_MARKER in normalized else "fail"


def review_targets(agent: AgentConfig, outgoing: list[EdgeConfig], state: WorkflowState) -> list[str] | str:
    """Select revision or approval targets from the review outcome and wire labels."""
    if not outgoing:
        return END_NODE_ID
    decision = state.get("decisions", {}).get(agent.id, "approved")
    counts = state.get("review_counts", {}).get(agent.id, 0)
    revise_edges = _edges_with_labels(outgoing, REVIEW_CHANGE_LABELS)
    approve_edges = _edges_with_labels(outgoing, REVIEW_APPROVAL_LABELS)
    if decision == "revise":
        _ensure_retry_available(agent, counts)
        return [edge.target for edge in revise_edges] or [outgoing[0].target]
    return [edge.target for edge in approve_edges] or [outgoing[-1].target]


def test_targets(
    agent: AgentConfig,
    outgoing: list[EdgeConfig],
    state: WorkflowState,
    target_kinds: dict[str, str] | None = None,
) -> list[str] | str:
    """Select test-failure repair or passing continuation targets."""
    if not outgoing:
        return END_NODE_ID
    decision = state.get("decisions", {}).get(agent.id, "fail")
    failures = state.get("review_counts", {}).get(agent.id, 0)
    failure_edges = _edges_with_labels(outgoing, TEST_FAILURE_LABELS)
    pass_edges = _edges_with_labels(outgoing, TEST_SUCCESS_LABELS)
    if decision == "fail":
        _ensure_retry_available(agent, failures)
        if failure_edges:
            return [edge.target for edge in failure_edges]
        developer_edges = [edge for edge in outgoing if (target_kinds or {}).get(edge.target, "").casefold() == DEVELOPER_KIND]
        return [edge.target for edge in developer_edges] or END_NODE_ID
    if pass_edges:
        return [edge.target for edge in pass_edges]
    ordinary_edges = [edge for edge in outgoing if not _has_label_keyword(edge, TEST_FAILURE_LABELS)]
    non_developer_edges = [edge for edge in ordinary_edges if (target_kinds or {}).get(edge.target, "").casefold() != DEVELOPER_KIND]
    if len(non_developer_edges) == 1:
        return [non_developer_edges[0].target]
    if len(ordinary_edges) == 1:
        return [ordinary_edges[0].target]
    return END_NODE_ID


def routing_targets(
    agent: AgentConfig,
    outgoing: list[EdgeConfig],
    decision: str,
    counts: int,
    target_kinds: dict[str, str],
) -> list[str] | str:
    """Resolve one configured outcome wire or apply role-specific routing policy."""
    if agent.success_edge_id or agent.failure_edge_id:
        failed = decision in {"revise", "fail"}
        if failed:
            _ensure_retry_available(agent, counts)
        edge_id = agent.failure_edge_id if failed else agent.success_edge_id
        selected_edge = next((edge for edge in outgoing if edge.id == edge_id), None)
        return [selected_edge.target] if selected_edge else END_NODE_ID

    state: WorkflowState = {
        "requirement": "",
        "project_path": "",
        "messages": [],
        "decisions": {agent.id: decision},
        "review_counts": {agent.id: counts},
        "handoffs": {},
        "test_commands": {},
    }
    if agent.kind.casefold() in REVIEWER_KINDS:
        return review_targets(agent, outgoing, state)
    if agent.kind.casefold() in TESTER_KINDS:
        return test_targets(agent, outgoing, state, target_kinds)
    return [edge.target for edge in outgoing] or END_NODE_ID


def is_test_agent(agent: AgentConfig) -> bool:
    """Identify roles that require observed test-command evidence."""
    return agent.kind.casefold() in TESTER_KINDS


def latest_message_from_developer(state: WorkflowState, agent_kinds: dict[str, str]) -> bool:
    """Check whether the latest upstream result came from a Developer agent."""
    messages = state.get("messages", [])
    if not messages:
        return False
    latest_agent_id = messages[-1].get("agent_id", "")
    return agent_kinds.get(latest_agent_id, "").casefold() == DEVELOPER_KIND


def _ensure_retry_available(agent: AgentConfig, counts: int) -> None:
    """Stop failure cycles once this agent exceeds its configured retry allowance."""
    if counts > agent.max_retries:
        raise RuntimeError(f"{agent.name} reached its configured retry limit ({agent.max_retries}).")


def _edges_with_labels(outgoing: list[EdgeConfig], keywords: tuple[str, ...]) -> list[EdgeConfig]:
    return [edge for edge in outgoing if _has_label_keyword(edge, keywords)]


def _has_label_keyword(edge: EdgeConfig, keywords: tuple[str, ...]) -> bool:
    label = edge.label.casefold()
    return any(keyword in label for keyword in keywords)

"""Validate workflow topology and compile directed LangGraph edges."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from threadline_backend.core.constants import END_NODE_ID, REVIEWER_KINDS, TESTER_KINDS
from threadline_backend.domain.contracts import AgentConfig, EdgeConfig, RunRequest, WorkflowState
from threadline_backend.domain.routing import review_targets, routing_targets, test_targets
from threadline_backend.application.agent_execution import AgentExecutor


def compile_workflow_graph(
    request: RunRequest,
    project_root: Path,
    event_queue: asyncio.Queue[dict[str, Any]],
    *,
    model_factory: Callable[..., Any] | None = None,
    api_key_resolver: Callable[[str], str | None] | None = None,
    model_config_resolver: Callable[[str], dict[str, Any]],
) -> Any:
    """Validate a workflow request and return its compiled graph."""
    outgoing, target_kinds, roots = _validate_topology(request)
    graph = StateGraph(WorkflowState)
    executor = AgentExecutor(
        request,
        project_root,
        event_queue,
        outgoing,
        target_kinds,
        model_factory=model_factory,
        api_key_resolver=api_key_resolver,
        model_config_resolver=model_config_resolver,
    )
    for agent in request.agents:
        graph.add_node(agent.id, executor.node(agent))
    for root in roots:
        graph.add_edge(START, root)
    for agent in request.agents:
        _connect_agent(graph, agent, outgoing[agent.id], target_kinds)
    return graph.compile()


def _validate_topology(
    request: RunRequest,
) -> tuple[dict[str, list[EdgeConfig]], dict[str, str], list[str]]:
    """Validate agent references, configured outcome wires, and workflow roots."""
    agent_ids = [agent.id for agent in request.agents]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("Agent IDs must be unique.")
    if any(edge.source not in agent_ids or edge.target not in agent_ids for edge in request.edges):
        raise ValueError("Every connection must point to an agent in this workflow.")
    if not request.edges:
        raise ValueError("Connect the agents into a workflow before running it.")
    outgoing = {agent_id: [edge for edge in request.edges if edge.source == agent_id] for agent_id in agent_ids}
    _validate_outcome_wires(request.agents, outgoing)
    incoming = {edge.target for edge in request.edges}
    roots = [agent_id for agent_id in agent_ids if agent_id not in incoming]
    if not roots:
        raise ValueError("The workflow needs at least one starting agent outside a review loop.")
    return outgoing, {agent.id: agent.kind for agent in request.agents}, roots


def _validate_outcome_wires(agents: list[AgentConfig], outgoing: dict[str, list[EdgeConfig]]) -> None:
    """Reject outcome selections that are not distinct outgoing connections."""
    for agent in agents:
        outgoing_ids = {edge.id for edge in outgoing[agent.id]}
        selected_ids = (agent.success_edge_id, agent.failure_edge_id)
        if any(edge_id and edge_id not in outgoing_ids for edge_id in selected_ids):
            raise ValueError(f"Configured outcome edge for {agent.name} is not connected from that agent.")
        if agent.success_edge_id and agent.success_edge_id == agent.failure_edge_id:
            raise ValueError(f"{agent.name} must use different wires for success and failure.")


def _connect_agent(
    graph: StateGraph,
    agent: AgentConfig,
    links: list[EdgeConfig],
    target_kinds: dict[str, str],
) -> None:
    """Register direct or outcome-based outgoing edges for one agent."""
    if not links:
        graph.add_edge(agent.id, END)
        return
    if agent.success_edge_id or agent.failure_edge_id or agent.kind.casefold() in REVIEWER_KINDS | TESTER_KINDS:
        targets = {edge.target for edge in links}
        graph.add_conditional_edges(
            agent.id,
            lambda state, current=agent, current_links=links: _select_targets(current, current_links, state, target_kinds),
            {**{target: target for target in targets}, END: END},
        )
        return
    for edge in links:
        graph.add_edge(agent.id, edge.target)


def _select_targets(
    agent: AgentConfig,
    links: list[EdgeConfig],
    state: WorkflowState,
    target_kinds: dict[str, str],
) -> list[str] | str:
    """Resolve graph destinations from the agent's recorded outcome."""
    decision = state.get("decisions", {}).get(agent.id, "fail")
    retries = state.get("review_counts", {}).get(agent.id, 0)
    if agent.success_edge_id or agent.failure_edge_id:
        return routing_targets(agent, links, decision, retries, target_kinds)
    kind = agent.kind.casefold()
    if kind in REVIEWER_KINDS:
        return review_targets(agent, links, state)
    if kind in TESTER_KINDS:
        return test_targets(agent, links, state, target_kinds)
    return [edge.target for edge in links] or END_NODE_ID

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from settings_store import model_settings
from threadline_backend.domain.contracts import AgentConfig, EdgeConfig, RunRequest, WorkflowState
from threadline_backend.application.agent_execution import tool_activity_detail as _tool_activity_detail
from threadline_backend.application.workflow_graph import compile_workflow_graph
from threadline_backend.domain.routing import (
    agent_result_decision as _agent_result_decision,
    is_test_agent as _is_test_agent,
    latest_message_from_developer as _latest_message_from_developer,
    review_targets as _review_targets,
    routing_targets as _routing_targets,
    test_targets as _test_targets,
)
from threadline_backend.infrastructure.model_provider import (
    create_chat_model,
    get_model_api_key as _get_model_api_key,
    get_model_config as _get_model_config,
    is_foundry_endpoint as _provider_is_foundry_endpoint,
    is_model_deployment_configured as _is_model_deployment_configured,
    missing_model_settings as _missing_model_settings,
    missing_settings_for_model as _missing_settings_for_model,
    required_model_settings,
    required_provider_keys,
)
from threadline_backend.infrastructure.command_tools import (
    is_test_command as _is_test_command,
    launch_command as _launch_command,
    project_python as _project_python,
    test_exit_code as _test_exit_code,
)
from threadline_backend.infrastructure.project_paths import resolve_project_path, safe_project_file
from threadline_backend.infrastructure.project_tools import build_project_tools as _project_tools

def get_model_config(model_id: str) -> dict[str, str]:
    """Load model metadata through the configured model settings store."""
    return _get_model_config(model_id, model_settings)


def get_model_api_key(model_id: str) -> str | None:
    """Resolve one model API key through the configured keyring store."""
    return _get_model_api_key(model_id, model_settings)


def _is_foundry_endpoint(endpoint: str | None) -> bool:
    """Preserve the legacy endpoint helper through the provider adapter."""
    return _provider_is_foundry_endpoint(endpoint)


def missing_settings_for_model(model_id: str) -> list[str]:
    """List required settings missing from one model configuration."""
    return _missing_settings_for_model(model_id, model_settings)


def is_model_deployment_configured(model_id: str) -> bool:
    """Check readiness for one configured deployment."""
    return _is_model_deployment_configured(model_id, model_settings)


def missing_model_settings(agents: list[AgentConfig]) -> list[str]:
    """Collect required settings missing across a workflow's models."""
    return _missing_model_settings(agents, model_settings)


def build_workflow(
    request: RunRequest,
    project_root: Path,
    event_queue: asyncio.Queue[dict[str, Any]],
    *,
    model_factory: Callable[..., Any] | None = None,
    api_key_resolver: Callable[[str], str | None] = get_model_api_key,
):
    """Compile a workflow using the layered application graph builder."""
    return compile_workflow_graph(
        request,
        project_root,
        event_queue,
        model_factory=model_factory,
        api_key_resolver=api_key_resolver,
        model_config_resolver=get_model_config,
    )
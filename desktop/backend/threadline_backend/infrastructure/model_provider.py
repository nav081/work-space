"""Azure model lookup, readiness checks, and LangChain model construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol
from urllib.parse import urlparse

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

from threadline_backend.core.constants import FOUNDRY_ENDPOINT_SUFFIX, FOUNDRY_OPENAI_API_PATH
from threadline_backend.domain.contracts import AgentConfig


class ModelSettingsRepository(Protocol):
    """Settings operations required by the model-provider adapter."""

    def get_model(self, model_id: str) -> dict[str, Any]: ...

    def get_api_key(self, model_id: str) -> str | None: ...

    def missing_settings(self, model_ids: set[str]) -> list[str]: ...


def get_model_config(model_id: str, settings: ModelSettingsRepository) -> dict[str, Any]:
    """Load one configured model or return a user-facing unknown-model error."""
    try:
        return settings.get_model(model_id)
    except ValueError as error:
        raise ValueError(f"Unknown model '{model_id}'. Refresh the model list and choose a configured model.") from error


def get_model_api_key(model_id: str, settings: ModelSettingsRepository) -> str | None:
    """Resolve a model secret through the settings repository."""
    return settings.get_api_key(model_id)


def required_provider_keys(agents: Sequence[AgentConfig]) -> set[str]:
    """Return model identifiers required by a workflow."""
    return {agent.model for agent in agents}


def required_model_settings(agents: Sequence[AgentConfig]) -> set[str]:
    """Return all model identifiers needed before execution."""
    return required_provider_keys(agents)


def missing_settings_for_model(model_id: str, settings: ModelSettingsRepository) -> list[str]:
    """List incomplete settings for one configured model."""
    settings.get_model(model_id)
    return settings.missing_settings({model_id})


def is_model_deployment_configured(model_id: str, settings: ModelSettingsRepository) -> bool:
    """Check whether a model has all required connection settings."""
    return not missing_settings_for_model(model_id, settings)


def missing_model_settings(agents: Sequence[AgentConfig], settings: ModelSettingsRepository) -> list[str]:
    """Collect incomplete settings for each distinct workflow model."""
    required_models = required_provider_keys(agents)
    return sorted({item for model_id in required_models for item in missing_settings_for_model(model_id, settings)})


def is_foundry_endpoint(endpoint: str | None) -> bool:
    """Identify Azure AI Foundry endpoints that use the v1 API path."""
    hostname = urlparse(endpoint or "").hostname or ""
    return hostname.casefold().endswith(FOUNDRY_ENDPOINT_SUFFIX)


def create_chat_model(model_config: dict[str, str], api_key: str) -> Any:
    """Build the LangChain chat model for the configured Azure endpoint type."""
    endpoint = model_config["endpoint"]
    deployment = model_config["deployment"]
    if is_foundry_endpoint(endpoint):
        base_url = (endpoint or "").rstrip("/")
        if not base_url.endswith(FOUNDRY_OPENAI_API_PATH):
            base_url = f"{base_url}{FOUNDRY_OPENAI_API_PATH}"
        return ChatOpenAI(model=deployment, base_url=f"{base_url}/", api_key=SecretStr(api_key))
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_version=model_config["api_version"],
        api_key=SecretStr(api_key),
        model=model_config["model"],
    )

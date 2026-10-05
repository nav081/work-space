from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import keyring
from dotenv.parser import parse_stream
from platformdirs import user_config_path
from pydantic import BaseModel, Field


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = BACKEND_DIR / "models.json"
KEYRING_SERVICE = "Threadline Azure OpenAI"
MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{3,100}$")


class ModelSettingsInput(BaseModel):
    id: str | None = Field(default=None, min_length=3, max_length=100)
    name: str = Field(min_length=1, max_length=120)
    model: str = Field(min_length=1, max_length=120)
    deployment: str = Field(min_length=1, max_length=160)
    endpoint: str = Field(min_length=1, max_length=500)
    api_version: str = Field(default="", max_length=80)
    api_key: str = Field(default="", max_length=4096)


def _is_foundry_endpoint(endpoint: str) -> bool:
    hostname = urlparse(endpoint).hostname or ""
    return hostname.casefold().endswith(".services.ai.azure.com")


class ModelSettingsStore:
    def __init__(self, config_path: Path | None = None, keyring_backend: Any = keyring, import_legacy: bool = True):
        self.config_path = config_path or user_config_path("Threadline", appauthor=False) / "models.json"
        self.keyring = keyring_backend
        self.import_legacy = import_legacy
        self.models = self._load()

    def _read_legacy_setting(self, name: str) -> str | None:
        environment_value = os.environ.get(name)
        if environment_value:
            return environment_value
        env_file = BACKEND_DIR / ".env"
        if not env_file.is_file():
            return None
        with env_file.open(encoding="utf-8") as stream:
            for binding in parse_stream(stream):
                if binding.key == name:
                    return binding.value
        return None

    def _load(self) -> list[dict[str, Any]]:
        if self.config_path.is_file():
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            return data.get("models", [])

        defaults = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
        models = []
        for item in defaults:
            model = {
                "id": item["id"],
                "name": item["name"],
                "provider": "azure_openai",
                "model": item["model"],
                "deployment": self._read_legacy_setting(item["deployment_env"]) if self.import_legacy else "",
                "endpoint": self._read_legacy_setting(item["endpoint_env"]) if self.import_legacy else "",
                "api_version": self._read_legacy_setting(item["api_version_env"]) if self.import_legacy else "",
            }
            legacy_key = self._read_legacy_setting(item["api_key_env"]) if self.import_legacy else None
            if legacy_key:
                self.keyring.set_password(KEYRING_SERVICE, model["id"], legacy_key)
            models.append(model)
        self.models = models
        self._persist()
        return models

    def _persist(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.config_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps({"models": self.models}, indent=2), encoding="utf-8")
        os.replace(temporary_path, self.config_path)

    def _find(self, model_id: str) -> dict[str, Any]:
        for model in self.models:
            if model["id"] == model_id:
                return model
        raise ValueError(f"Unknown model '{model_id}'.")

    def get_model(self, model_id: str) -> dict[str, Any]:
        return dict(self._find(model_id))

    def get_api_key(self, model_id: str) -> str | None:
        self._find(model_id)
        return self.keyring.get_password(KEYRING_SERVICE, model_id)

    def _public_model(self, model: dict[str, Any]) -> dict[str, Any]:
        api_key_configured = bool(self.keyring.get_password(KEYRING_SERVICE, model["id"]))
        endpoint_configured = bool(model["endpoint"])
        deployment_configured = bool(model["deployment"])
        version_configured = _is_foundry_endpoint(model["endpoint"]) or bool(model["api_version"])
        return {
            **model,
            "api_key_configured": api_key_configured,
            "configured": api_key_configured and endpoint_configured and deployment_configured and version_configured,
        }

    def list_models(self) -> list[dict[str, Any]]:
        return [self._public_model(model) for model in self.models]

    def missing_settings(self, model_ids: set[str]) -> list[str]:
        missing: list[str] = []
        for model_id in sorted(model_ids):
            model = self._find(model_id)
            if not self.keyring.get_password(KEYRING_SERVICE, model_id):
                missing.append(f"{model['name']}: API key")
            if not model["endpoint"]:
                missing.append(f"{model['name']}: endpoint")
            if not model["deployment"]:
                missing.append(f"{model['name']}: deployment name")
            if not _is_foundry_endpoint(model["endpoint"]) and not model["api_version"]:
                missing.append(f"{model['name']}: API version")
        return missing

    def save(self, payload: ModelSettingsInput) -> dict[str, Any]:
        name = payload.name.strip()
        model_name = payload.model.strip()
        deployment = payload.deployment.strip()
        endpoint = payload.endpoint.strip().rstrip("/")
        api_version = payload.api_version.strip()
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme != "https" or not parsed_endpoint.hostname:
            raise ValueError("Azure endpoint must be a valid HTTPS URL.")
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", model_name):
            raise ValueError("Model ID can contain letters, numbers, dots, underscores, and hyphens only.")
        model_id = payload.id or f"azure-custom-{uuid4().hex[:12]}"
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("Model identifier must contain 3 to 100 letters, numbers, dots, underscores, or hyphens.")
        existing = next((item for item in self.models if item["id"] == model_id), None)
        if existing is None and not payload.api_key:
            raise ValueError("Enter an API key for a new Azure model.")

        updated = {
            "id": model_id,
            "name": name,
            "provider": "azure_openai",
            "model": model_name,
            "deployment": deployment,
            "endpoint": endpoint,
            "api_version": api_version,
        }
        if payload.api_key:
            self.keyring.set_password(KEYRING_SERVICE, model_id, payload.api_key)
        if existing is None:
            self.models.append(updated)
        else:
            self.models[self.models.index(existing)] = updated
        self._persist()
        return self._public_model(updated)

    def delete(self, model_id: str) -> None:
        model = self._find(model_id)
        if self.keyring.get_password(KEYRING_SERVICE, model_id):
            self.keyring.delete_password(KEYRING_SERVICE, model_id)
        self.models.remove(model)
        self._persist()


model_settings = ModelSettingsStore()
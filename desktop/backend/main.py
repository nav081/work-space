from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI as FastAPIApplication, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from run_store import RunStore
from settings_store import ModelSettingsInput, model_settings
from workflow import RunRequest, build_workflow, missing_model_settings, resolve_project_path
from threadline_backend.application.workflow_run import WorkflowRunService
from threadline_backend.core.constants import (
    API_TITLE,
    API_VERSION,
    CORS_ALLOWED_HEADERS,
    CORS_ALLOWED_METHODS,
    CORS_ALLOWED_ORIGINS,
    DEFAULT_RUN_HISTORY_LIMIT,
    NO_CACHE_HEADER,
    NO_CACHE_VALUE,
    PROJECT_PATH_REQUEST_MAX_LENGTH,
    PROXY_BUFFERING_DISABLED,
    PROXY_BUFFERING_HEADER,
    SSE_MEDIA_TYPE,
)
from threadline_backend.presentation.serializers import encode_sse_event

app = FastAPIApplication(title=API_TITLE, version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)


class ProjectRequest(BaseModel):
    """Project directory path submitted by the desktop editor."""

    path: str = Field(min_length=1, max_length=PROJECT_PATH_REQUEST_MAX_LENGTH)


run_store = RunStore()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Report local API readiness."""
    return {"status": "ok"}


@app.get("/api/models")
async def models() -> list[dict[str, Any]]:
    """List configured model deployments for the desktop client."""
    return model_settings.list_models()


@app.get("/api/settings/models")
async def settings_models() -> list[dict[str, Any]]:
    """List model settings for the Settings screen."""
    return model_settings.list_models()


@app.post("/api/settings/models")
async def save_model_settings(request: ModelSettingsInput) -> dict[str, Any]:
    """Validate and persist model connection details and credentials."""
    try:
        return model_settings.save(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Windows Credential Manager could not save this API key.") from error


@app.delete("/api/settings/models/{model_id}")
async def delete_model_settings(model_id: str) -> dict[str, str]:
    """Delete a model configuration and its stored credential."""
    try:
        model_settings.delete(model_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Windows Credential Manager could not remove this API key.") from error
    return {"status": "deleted"}


@app.post("/api/projects/validate")
async def validate_project(request: ProjectRequest) -> dict[str, str]:
    """Resolve a project directory before saving it in the workflow editor."""
    try:
        root = resolve_project_path(request.path)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"path": str(root), "name": root.name or str(root)}


@app.get("/api/runs")
async def list_runs(limit: int = DEFAULT_RUN_HISTORY_LIMIT) -> list[dict[str, Any]]:
    """Return recent durable workflow summaries."""
    return run_store.list_runs(limit)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Return one durable run and its full event history."""
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@app.post("/api/workflows/run")
async def run_workflow(request: RunRequest) -> StreamingResponse:
    """Validate a saved workflow and stream its persisted execution events."""
    try:
        root = resolve_project_path(request.project_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        missing = missing_model_settings(request.agents)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if missing:
        raise HTTPException(status_code=400, detail=f"Configure these Azure OpenAI settings in app Settings: {', '.join(missing)}")
    run_id = run_store.create_run(root.name or str(root), str(root), request.requirement, request.model_dump(mode="json"))
    service = WorkflowRunService(run_store, build_workflow)
    events = _sse_events(service, request, root, run_id)
    return StreamingResponse(events, media_type=SSE_MEDIA_TYPE, headers={NO_CACHE_HEADER: NO_CACHE_VALUE, PROXY_BUFFERING_HEADER: PROXY_BUFFERING_DISABLED})


async def _sse_events(
    service: WorkflowRunService,
    request: RunRequest,
    root: Path,
    run_id: str,
) -> AsyncIterator[str]:
    """Adapt application event dictionaries to SSE frames."""
    async for payload in service.stream_events(request, root, run_id):
        yield encode_sse_event(payload)
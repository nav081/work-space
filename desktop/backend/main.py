from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI as FastAPIApplication, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from run_store import RunStore
from settings_store import ModelSettingsInput, model_settings
from workflow import RunRequest, WorkflowState, build_workflow, missing_model_settings, resolve_project_path

app = FastAPIApplication(title="Threadline Local Agent Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "tauri://localhost", "http://tauri.localhost"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ProjectRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)


run_store = RunStore()


def _event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/models")
async def models() -> list[dict[str, Any]]:
    return model_settings.list_models()


@app.get("/api/settings/models")
async def settings_models() -> list[dict[str, Any]]:
    return model_settings.list_models()


@app.post("/api/settings/models")
async def save_model_settings(request: ModelSettingsInput) -> dict[str, Any]:
    try:
        return model_settings.save(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Windows Credential Manager could not save this API key.") from error


@app.delete("/api/settings/models/{model_id}")
async def delete_model_settings(model_id: str) -> dict[str, str]:
    try:
        model_settings.delete(model_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Windows Credential Manager could not remove this API key.") from error
    return {"status": "deleted"}


@app.post("/api/projects/validate")
async def validate_project(request: ProjectRequest) -> dict[str, str]:
    try:
        root = resolve_project_path(request.path)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"path": str(root), "name": root.name or str(root)}


@app.get("/api/runs")
async def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    return run_store.list_runs(limit)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@app.post("/api/workflows/run")
async def run_workflow(request: RunRequest) -> StreamingResponse:
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

    async def events() -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        run_status = "running"

        async def execute() -> None:
            try:
                graph = build_workflow(request, root, queue)
                await queue.put({"type": "run_started", "run_id": run_id, "project": root.name, "agent_count": len(request.agents)})
                initial_state: WorkflowState = {"requirement": request.requirement, "project_path": str(root), "messages": [], "decisions": {}, "review_counts": {}, "handoffs": {}, "test_commands": {}}
                async for update in graph.astream(initial_state, config={"recursion_limit": 1000, "max_concurrency": 8}, stream_mode="updates"):
                    for agent_id, result in update.items():
                        messages = result.get("messages", []) if isinstance(result, dict) else []
                        for message in messages:
                            await queue.put({"type": "agent_completed", **message})
                await queue.put({"type": "run_completed"})
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await queue.put({"type": "run_error", "message": str(error)[:1000]})
            finally:
                await queue.put({"type": "stream_end"})

        task = asyncio.create_task(execute())
        try:
            while True:
                payload = await queue.get()
                if payload["type"] == "stream_end":
                    break
                if payload["type"] == "run_completed":
                    run_status = "completed"
                    run_store.finish_run(run_id, run_status)
                elif payload["type"] == "run_error":
                    run_status = "failed"
                    run_store.finish_run(run_id, run_status)
                run_store.append_event(run_id, payload)
                yield _event(payload)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if run_status == "running":
                run_status = "interrupted"
            run_store.finish_run(run_id, run_status)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
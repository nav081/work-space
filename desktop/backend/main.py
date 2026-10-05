from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from workflow import MODEL_CATALOG, RunRequest, build_workflow, is_model_deployment_configured, missing_model_settings, resolve_project_path

app = FastAPI(title="Threadline Local Agent Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "tauri://localhost", "http://tauri.localhost"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ProjectRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)


def _event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/models")
async def models() -> list[dict[str, Any]]:
    return [
        {"id": model["id"], "name": model["name"], "provider": model["provider"], "configured": is_model_deployment_configured(model["id"])}
        for model in MODEL_CATALOG
    ]


@app.post("/api/projects/validate")
async def validate_project(request: ProjectRequest) -> dict[str, str]:
    try:
        root = resolve_project_path(request.path)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"path": str(root), "name": root.name or str(root)}


@app.post("/api/workflows/run")
async def run_workflow(request: RunRequest) -> StreamingResponse:
    try:
        root = resolve_project_path(request.project_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    missing = missing_model_settings(request.agents)
    if missing:
        raise HTTPException(status_code=400, detail=f"Configure these Azure OpenAI settings in backend/.env: {', '.join(missing)}")

    async def events() -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def execute() -> None:
            try:
                graph = build_workflow(request, root, queue)
                await queue.put({"type": "run_started", "project": root.name, "agent_count": len(request.agents)})
                initial_state = {"requirement": request.requirement, "project_path": str(root), "messages": [], "decisions": {}, "review_counts": {}}
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
                yield _event(payload)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
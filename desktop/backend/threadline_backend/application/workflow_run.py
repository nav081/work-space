"""Coordinate workflow graph execution, event persistence, and final run status."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Protocol

from threadline_backend.core.constants import (
    COMPLETED_STATUS,
    FAILED_STATUS,
    GRAPH_MAX_CONCURRENCY,
    GRAPH_RECURSION_LIMIT,
    INTERRUPTED_STATUS,
    RUN_ERROR_MESSAGE_LIMIT,
    RUNNING_STATUS,
    STREAM_END_EVENT_TYPE,
)
from threadline_backend.domain.contracts import RunRequest, WorkflowState


class RunRepository(Protocol):
    """Persistence operations required by workflow execution."""

    def append_event(self, run_id: str, payload: dict[str, Any]) -> None: ...

    def finish_run(self, run_id: str, status: str) -> None: ...


class WorkflowRunService:
    """Stream workflow events while recording durable run state."""

    def __init__(self, repository: RunRepository, graph_builder: Callable[..., Any]):
        self.repository = repository
        self.graph_builder = graph_builder

    async def stream_events(self, request: RunRequest, root: Path, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Execute a workflow and yield each event after persisting it."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        status = RUNNING_STATUS
        task = asyncio.create_task(self._execute(request, root, run_id, queue))
        try:
            while True:
                payload = await queue.get()
                event_type = payload["type"]
                if event_type == STREAM_END_EVENT_TYPE:
                    break
                if event_type == "run_completed":
                    status = COMPLETED_STATUS
                    self.repository.finish_run(run_id, status)
                elif event_type == "run_error":
                    status = FAILED_STATUS
                    self.repository.finish_run(run_id, status)
                self.repository.append_event(run_id, payload)
                yield payload
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if status == RUNNING_STATUS:
                status = INTERRUPTED_STATUS
            self.repository.finish_run(run_id, status)

    async def _execute(
        self,
        request: RunRequest,
        root: Path,
        run_id: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Run the graph and emit terminal lifecycle events."""
        try:
            graph = self.graph_builder(request, root, queue)
            await queue.put({"type": "run_started", "run_id": run_id, "project": root.name, "agent_count": len(request.agents)})
            initial_state: WorkflowState = {
                "requirement": request.requirement,
                "project_path": str(root),
                "messages": [],
                "decisions": {},
                "review_counts": {},
                "handoffs": {},
                "test_commands": {},
            }
            async for update in graph.astream(
                initial_state,
                config={"recursion_limit": GRAPH_RECURSION_LIMIT, "max_concurrency": GRAPH_MAX_CONCURRENCY},
                stream_mode="updates",
            ):
                for result in update.values():
                    messages = result.get("messages", []) if isinstance(result, dict) else []
                    for message in messages:
                        await queue.put({"type": "agent_completed", **message})
            await queue.put({"type": "run_completed"})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await queue.put({"type": "run_error", "message": str(error)[:RUN_ERROR_MESSAGE_LIMIT]})
        finally:
            await queue.put({"type": STREAM_END_EVENT_TYPE})

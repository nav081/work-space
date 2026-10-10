from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from platformdirs import user_data_path
from threadline_backend.core.constants import (
    DEFAULT_RUN_EVENT_TYPE,
    DEFAULT_RUN_HISTORY_LIMIT,
    FINAL_RUN_STATUSES,
    MAX_RUN_HISTORY_LIMIT,
    MIN_RUN_HISTORY_LIMIT,
    RUN_DATABASE_FILENAME,
    RUNNING_STATUS,
    SQLITE_CONNECT_TIMEOUT_SECONDS,
    TIMESTAMP_PRECISION,
)
from threadline_backend.presentation.run_serializers import serialize_run_detail, serialize_run_summary


def _timestamp() -> str:
    """Return a UTC timestamp suitable for durable run events."""
    return datetime.now(UTC).isoformat(timespec=TIMESTAMP_PRECISION)


class RunStore:
    """Persist workflow runs and their streamed event history in SQLite."""

    def __init__(self, database_path: Path | None = None):
        """Initialize a run repository at the explicit or per-user database path."""
        self.database_path = database_path or user_data_path("Threadline", appauthor=False) / RUN_DATABASE_FILENAME
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a configured SQLite connection for one repository operation."""
        connection = sqlite3.connect(self.database_path, timeout=SQLITE_CONNECT_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        """Create run and event tables and their lookup indexes when needed."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    project_path TEXT NOT NULL,
                    requirement TEXT NOT NULL,
                    workflow_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS run_events_by_run ON run_events(run_id, event_id);
                CREATE INDEX IF NOT EXISTS runs_by_started ON runs(started_at DESC);
                """
            )

    def create_run(self, project_name: str, project_path: str, requirement: str, workflow: dict[str, Any]) -> str:
        """Persist a new running workflow snapshot and return its identifier."""
        run_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs (run_id, started_at, status, project_name, project_path, requirement, workflow_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, _timestamp(), RUNNING_STATUS, project_name, project_path, requirement, json.dumps(workflow, ensure_ascii=True)),
            )
        return run_id

    def append_event(self, run_id: str, payload: dict[str, Any]) -> None:
        """Append one serialized event to a workflow run."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO run_events (run_id, created_at, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (run_id, _timestamp(), str(payload.get("type", DEFAULT_RUN_EVENT_TYPE)), json.dumps(payload, ensure_ascii=True)),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        """Set a run's terminal status and completion time."""
        if status not in FINAL_RUN_STATUSES:
            raise ValueError("Invalid final run status.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (status, _timestamp(), run_id),
            )

    def list_runs(self, limit: int = DEFAULT_RUN_HISTORY_LIMIT) -> list[dict[str, Any]]:
        """Return recent run summaries within the supported result limit."""
        bounded_limit = min(max(limit, MIN_RUN_HISTORY_LIMIT), MAX_RUN_HISTORY_LIMIT)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, started_at, completed_at, status, project_name, project_path, requirement, workflow_json "
                "FROM runs ORDER BY started_at DESC LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return [serialize_run_summary(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a run snapshot and all events, or None when it does not exist."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            events = connection.execute(
                "SELECT created_at, event_type, payload_json FROM run_events WHERE run_id = ? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        return serialize_run_detail(row, events)
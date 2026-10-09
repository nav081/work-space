from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from platformdirs import user_data_path


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class RunStore:
    def __init__(self, database_path: Path | None = None):
        self.database_path = database_path or user_data_path("Threadline", appauthor=False) / "runs.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
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
        run_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs (run_id, started_at, status, project_name, project_path, requirement, workflow_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, _timestamp(), "running", project_name, project_path, requirement, json.dumps(workflow, ensure_ascii=True)),
            )
        return run_id

    def append_event(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO run_events (run_id, created_at, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (run_id, _timestamp(), str(payload.get("type", "event")), json.dumps(payload, ensure_ascii=True)),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("Invalid final run status.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (status, _timestamp(), run_id),
            )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        bounded_limit = min(max(limit, 1), 200)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, started_at, completed_at, status, project_name, project_path, requirement, workflow_json "
                "FROM runs ORDER BY started_at DESC LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return [self._summary(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            events = connection.execute(
                "SELECT created_at, event_type, payload_json FROM run_events WHERE run_id = ? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        return {
            **self._summary(row),
            "workflow": json.loads(row["workflow_json"]),
            "events": [
                {"created_at": event["created_at"], **json.loads(event["payload_json"])}
                for event in events
            ],
        }

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        workflow = json.loads(row["workflow_json"])
        agents = workflow.get("agents", [])
        return {
            "run_id": row["run_id"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "status": row["status"],
            "project_name": row["project_name"],
            "project_path": row["project_path"],
            "requirement": row["requirement"],
            "agent_count": len(agents),
            "agents": [{"id": agent.get("id"), "name": agent.get("name"), "kind": agent.get("kind")} for agent in agents],
        }
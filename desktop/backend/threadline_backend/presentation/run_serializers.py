"""Serialize persisted run records for history API responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def serialize_run_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one database run row into the public history summary shape."""
    workflow = _decode_json_object(row["workflow_json"])
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


def serialize_run_detail(row: Mapping[str, Any], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert a database run row and event rows into the full run response."""
    return {
        **serialize_run_summary(row),
        "workflow": _decode_json_object(row["workflow_json"]),
        "events": [
            {"created_at": event["created_at"], **_decode_json_object(event["payload_json"])}
            for event in events
        ],
    }


def _decode_json_object(value: str) -> dict[str, Any]:
    """Decode a JSON object persisted by the run repository."""
    decoded = json.loads(value)
    return decoded if isinstance(decoded, dict) else {}

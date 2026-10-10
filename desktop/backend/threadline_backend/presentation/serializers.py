"""Serialize workflow events for the Server-Sent Events transport."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def encode_sse_event(payload: Mapping[str, Any]) -> str:
    """Encode one event payload as an SSE data frame."""
    return f"data: {json.dumps(dict(payload), ensure_ascii=True)}\n\n"

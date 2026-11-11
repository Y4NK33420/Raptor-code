"""Server-sent events helpers (plain ``StreamingResponse``, no extra dependency)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from fastapi.responses import StreamingResponse

Event = tuple[str, Any]


def format_event(event: str, data: Any, event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False, default=str)
    lines.extend(f"data: {line}" for line in payload.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


def sse_response(events: Iterable[Event | str]) -> StreamingResponse:
    """Stream ``(event, data)`` tuples (or preformatted strings) as ``text/event-stream``."""

    def body() -> Iterator[str]:
        try:
            for item in events:
                yield item if isinstance(item, str) else format_event(*item)
        except Exception as exc:  # surface failures to the client instead of a broken stream
            yield format_event("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

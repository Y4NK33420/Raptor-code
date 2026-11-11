"""Background jobs with a replayable event log (streamed to clients over SSE)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .sse import format_event

logger = logging.getLogger(__name__)

TERMINAL = frozenset({"done", "error"})


@dataclass
class Job:
    id: str
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def emit(self, event: str, data: dict[str, Any]) -> None:
        start = self.started_at or self.created_at
        payload = {**data, "t": round(time.time() - start, 3)}
        with self._cond:
            self.events.append((event, payload))
            self._cond.notify_all()

    def _set_status(self, status: str) -> None:
        with self._cond:
            self.status = status
            self._cond.notify_all()

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "meta": self.meta,
            "created_at": self.created_at,
            "result": self.result,
            "error": self.error,
            "n_events": len(self.events),
        }

    def stream(self, after: int = -1, keepalive: float = 10.0) -> Iterator[str]:
        """Replay events with id > ``after``, then follow live events until the job ends."""
        pos = max(after + 1, 0)
        while True:
            with self._cond:
                start = pos
                self._cond.wait_for(
                    lambda start=start: len(self.events) > start or self.finished,  # type: ignore[misc]
                    keepalive,
                )
                batch = self.events[pos:]
                finished = self.finished
            if not batch and not finished:
                yield ": keepalive\n\n"
                continue
            for event, data in batch:
                yield format_event(event, data, pos)
                pos += 1
            if finished and pos >= len(self.events):
                return


class JobManager:
    """Runs jobs one at a time (builds are heavy and share one API quota)."""

    def __init__(self, workers: int = 1, keep: int = 50) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raptor-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.keep = keep

    def submit(
        self, kind: str, fn: Callable[[Job], dict[str, Any]], meta: dict[str, Any] | None = None
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, meta=meta or {})
        with self._lock:
            self._jobs[job.id] = job
            self._trim()
        job.emit("queued", {"stage": "queued"})
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], dict[str, Any]]) -> None:
        job.started_at = time.time()
        job._set_status("running")
        try:
            job.result = fn(job)
        except Exception as exc:
            logger.exception("job %s failed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"
            job.emit("error", {"stage": "error", "message": job.error})
            job._set_status("error")
            return
        job.emit("complete", {"stage": "complete", **(job.result or {})})
        job._set_status("done")

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id]

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: -j.created_at)

    def _trim(self) -> None:
        done = [j for j in self._jobs.values() if j.finished]
        for j in sorted(done, key=lambda j: j.created_at)[: max(0, len(self._jobs) - self.keep)]:
            self._jobs.pop(j.id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

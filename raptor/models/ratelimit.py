"""A thread-safe requests-per-minute limiter shared by all callers of one API bucket."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Sliding-window limiter: at most ``rpm`` acquisitions in any 60 s window."""

    def __init__(self, rpm: int | None) -> None:
        self.rpm = rpm if rpm and rpm > 0 else None
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rpm is None:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0])
            time.sleep(max(wait, 0.01))


_SHARED: dict[str, RateLimiter] = {}
_SHARED_LOCK = threading.Lock()


def shared_limiter(name: str, rpm: int | None) -> RateLimiter:
    """Return the process-wide limiter for ``name`` (created on first use)."""
    with _SHARED_LOCK:
        limiter = _SHARED.get(name)
        if limiter is None or limiter.rpm != (rpm if rpm and rpm > 0 else None):
            limiter = RateLimiter(rpm)
            _SHARED[name] = limiter
        return limiter

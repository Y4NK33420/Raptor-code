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

    def try_acquire(self, cost: int = 1) -> float:
        """Take ``cost`` slots if free. Returns 0.0 on success, else seconds until they free up.

        A cost above ``rpm`` is clamped to ``rpm`` (it waits for an empty window).
        """
        if self.rpm is None:
            return 0.0
        cost = max(1, min(cost, self.rpm))
        with self._lock:
            now = time.monotonic()
            while self._times and now - self._times[0] >= 60.0:
                self._times.popleft()
            if len(self._times) + cost <= self.rpm:
                self._times.extend([now] * cost)
                return 0.0
            # The entry that must expire before ``cost`` slots are free.
            blocker = self._times[len(self._times) + cost - self.rpm - 1]
            return max(60.0 - (now - blocker), 0.01)

    def acquire(self) -> None:
        while (wait := self.try_acquire()) > 0:
            time.sleep(wait)


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

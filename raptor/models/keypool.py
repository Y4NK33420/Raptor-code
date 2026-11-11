"""Spread Gemini calls over several API keys, with per-key rate limits and failover.

Configure extra keys as ``GEMINI_API_KEY2``, ``GEMINI_API_KEY3``, ... Each key gets its
own requests-per-minute budget per model. Calls go round-robin to the first key with
budget left. When a key returns 429 it is cooled down for that model (one minute, or one
hour for a per-day quota) and the call is retried on the next key.

Note: Google applies quotas per Cloud *project*, so keys from the same project share
one quota. Keys only add capacity if they belong to different projects.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .ratelimit import RateLimiter

logger = logging.getLogger(__name__)

T = TypeVar("T")

RATE_LIMIT_COOLDOWN = 60.0
DAILY_QUOTA_COOLDOWN = 3600.0
MAX_ATTEMPTS = 8


def is_rate_limit(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    return code == 429 or "resource_exhausted" in text or "429" in text


def is_retryable(exc: BaseException) -> bool:
    if is_rate_limit(exc):
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code >= 500:
        return True
    text = str(exc).lower()
    return "unavailable" in text or "deadline" in text or "timed out" in text


def _cooldown_for(exc: BaseException) -> float:
    text = str(exc).lower()
    return DAILY_QUOTA_COOLDOWN if "perday" in text or "per day" in text else RATE_LIMIT_COOLDOWN


@dataclass
class KeySlot:
    index: int
    client: Any
    limiters: dict[str, RateLimiter] = field(default_factory=dict)
    cooldown_until: dict[str, float] = field(default_factory=dict)


class KeyPool:
    def __init__(self, keys: Sequence[str], client_factory: Callable[[str], Any]) -> None:
        if not keys:
            raise ValueError("KeyPool needs at least one API key")
        self.slots = [KeySlot(i, client_factory(k)) for i, k in enumerate(keys)]
        self._cursor = 0
        self._lock = threading.Lock()
        self.calls_per_key = [0] * len(self.slots)

    def __len__(self) -> int:
        return len(self.slots)

    def _limiter(self, slot: KeySlot, bucket: str, rpm: int | None) -> RateLimiter:
        lim = slot.limiters.get(bucket)
        if lim is None or lim.rpm != (rpm if rpm and rpm > 0 else None):
            lim = slot.limiters[bucket] = RateLimiter(rpm)
        return lim

    def acquire(self, bucket: str, rpm: int | None) -> KeySlot:
        """Block until some key has budget for ``bucket``; return that key's slot."""
        while True:
            with self._lock:
                now = time.monotonic()
                waits = []
                n = len(self.slots)
                for step in range(n):
                    slot = self.slots[(self._cursor + step) % n]
                    cool = slot.cooldown_until.get(bucket, 0.0) - now
                    if cool > 0:
                        waits.append(cool)
                        continue
                    wait = self._limiter(slot, bucket, rpm).try_acquire()
                    if wait == 0.0:
                        self._cursor = (slot.index + 1) % n
                        self.calls_per_key[slot.index] += 1
                        return slot
                    waits.append(wait)
            time.sleep(max(min(waits), 0.01))

    def cooldown(self, slot: KeySlot, bucket: str, seconds: float) -> None:
        with self._lock:
            slot.cooldown_until[bucket] = time.monotonic() + seconds

    def call(self, bucket: str, rpm: int | None, fn: Callable[[Any], T]) -> T:
        """Run ``fn(client)`` on an available key, failing over and retrying as needed."""
        attempts = MAX_ATTEMPTS + len(self.slots)
        for attempt in range(attempts):
            slot = self.acquire(bucket, rpm)
            try:
                return fn(slot.client)
            except Exception as exc:
                if not is_retryable(exc) or attempt == attempts - 1:
                    raise
                if is_rate_limit(exc):
                    seconds = _cooldown_for(exc)
                    logger.warning(
                        "Gemini key #%d rate-limited on %s; cooling down %.0fs%s",
                        slot.index + 1,
                        bucket,
                        seconds,
                        " (trying next key)" if len(self.slots) > 1 else "",
                    )
                    self.cooldown(slot, bucket, seconds)
                else:
                    delay = min(60.0, 2.0 * 2**attempt) * random.uniform(0.5, 1.0)
                    logger.warning("Gemini call failed (%s); retrying in %.1fs", exc, delay)
                    time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover


_POOLS: dict[tuple[str, ...], KeyPool] = {}
_POOLS_LOCK = threading.Lock()


def shared_pool(keys: Sequence[str], client_factory: Callable[[str], Any]) -> KeyPool:
    """Process-wide pool for this exact set of keys, so all models share key budgets."""
    with _POOLS_LOCK:
        pool = _POOLS.get(tuple(keys))
        if pool is None:
            pool = _POOLS[tuple(keys)] = KeyPool(keys, client_factory)
        return pool

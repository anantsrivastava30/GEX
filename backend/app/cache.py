"""Tiny in-process TTL cache.

Kept behind a two-method interface so a Redis implementation can replace it
without touching routers or services. TTL guidance (see docs/UW_PARITY_PLAN.md):
quotes 15s · chains 60s · expirations 24h · macro 5min · news 10min.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

_MAX_ENTRIES = 2048


class TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get_or_compute(self, key: str, ttl: float, fn: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._store.get(key)
            if hit is not None and hit[1] > now:
                return hit[0]

        value = fn()

        with self._lock:
            if len(self._store) >= _MAX_ENTRIES:
                self._store = {
                    k: v for k, v in self._store.items() if v[1] > now
                }
            self._store[key] = (value, now + ttl)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


cache = TTLCache()

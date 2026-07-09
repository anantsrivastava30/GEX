"""Token-bucket guard shared by every Tradier call in the process."""

from __future__ import annotations

import threading
import time

from fastapi import HTTPException


class TokenBucket:
    def __init__(self, requests_per_minute: int) -> None:
        self.capacity = float(requests_per_minute)
        self.tokens = float(requests_per_minute)
        self.fill_rate = requests_per_minute / 60.0
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self.tokens = min(
                self.capacity, self.tokens + (now - self.updated) * self.fill_rate
            )
            self.updated = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def acquire_or_raise(self) -> None:
        if not self.try_acquire():
            raise HTTPException(
                status_code=429,
                detail="Upstream market-data rate limit reached; retry shortly.",
            )

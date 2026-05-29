"""Per-key token-bucket rate limiting (in-memory, monotonic-clock based)."""
from __future__ import annotations

from time import monotonic


class TokenBucket:
    def __init__(self, rate_per_min: int) -> None:
        self.capacity = float(rate_per_min)
        self.refill_per_sec = rate_per_min / 60.0
        self.tokens = float(rate_per_min)
        self.updated = monotonic()

    def allow(self, cost: float = 1.0) -> bool:
        now = monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_sec)
        self.updated = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class RateLimiter:
    def __init__(self, rate_per_min: int) -> None:
        self.rate = rate_per_min
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, key: str) -> bool:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = TokenBucket(self.rate)
        return bucket.allow()

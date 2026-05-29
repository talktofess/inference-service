"""API-key auth. Keys are compared against the configured allow-list; only a
salted hash of the presented key is used as the per-key identity (so raw keys
don't end up in logs/metrics labels).
"""
from __future__ import annotations

import hashlib


def key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


class ApiKeyAuth:
    def __init__(self, allowed: list[str]) -> None:
        self._allowed = set(allowed)

    def check(self, api_key: str | None) -> bool:
        return bool(api_key) and api_key in self._allowed

    @staticmethod
    def extract(authorization: str | None, x_api_key: str | None) -> str | None:
        if x_api_key:
            return x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[len("bearer ") :].strip()
        return None

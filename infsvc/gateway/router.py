"""Primary -> fallback routing.

Tries the primary backend; on health failure or exception, serves from the
fallback (e.g. a smaller/quantized model under load). Reports which backend
actually served so the gateway can label metrics.
"""
from __future__ import annotations

from ..backends.base import InferenceBackend
from ..schemas import GenerationRequest, GenerationResponse


class Router:
    def __init__(self, primary: InferenceBackend, fallback: InferenceBackend | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate(self, req: GenerationRequest) -> tuple[GenerationResponse, str]:
        try:
            if await self.primary.health():
                return await self.primary.generate(req), self.primary.name
        except Exception:
            pass
        if self.fallback is not None:
            return await self.fallback.generate(req), f"{self.fallback.name}(fallback)"
        # No fallback: surface the primary's behavior (and let it raise).
        return await self.primary.generate(req), self.primary.name

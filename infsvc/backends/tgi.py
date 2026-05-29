"""TGI (Text Generation Inference) backend — HTTP client to a TGI server.

Run TGI via its Docker image, then point at backend="tgi". Implemented enough
to benchmark against; exercised in the GPU phase.
"""
from __future__ import annotations

import time

from .. import settings
from ..schemas import GenerationRequest, GenerationResponse


class TGIBackend:
    name = "tgi"
    simulated = False

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (base_url or settings.TGI_URL).rstrip("/")
        self.profile = {"model": model, "engine": "tgi", "url": self.base_url}

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        import httpx

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/generate",
                json={
                    "inputs": req.prompt,
                    "parameters": {"max_new_tokens": req.max_tokens, "temperature": req.temperature or None},
                },
            )
            r.raise_for_status()
            data = r.json()
        total_ms = (time.perf_counter() - t0) * 1000.0
        text = data.get("generated_text", "")
        return GenerationResponse(
            text=text,
            prompt_tokens=len(req.prompt.split()),
            completion_tokens=len(text.split()),
            ttft_ms=total_ms,
            total_ms=total_ms,
        )

    async def stream(self, req: GenerationRequest):
        resp = await self.generate(req)
        yield resp.text

    async def health(self) -> bool:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

"""vLLM backend — an HTTP client to vLLM's OpenAI-compatible server.

Start the engine (on a Linux/WSL2 GPU box) with:
    python -m vllm.entrypoints.openai.api_server --model <hf-model> --port 8000

Then point the gateway/benchmark at backend="vllm". This client is real and
runnable; it just needs a reachable server, so it's exercised in the GPU phase.
"""
from __future__ import annotations

import time

from .. import settings
from ..schemas import GenerationRequest, GenerationResponse


class VLLMBackend:
    name = "vllm"
    simulated = False

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (base_url or settings.VLLM_URL).rstrip("/")
        self.profile = {"model": model, "dtype": "fp16", "engine": "vllm", "url": self.base_url}

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        import httpx

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/completions",
                json={
                    "model": self.model,
                    "prompt": req.prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                },
            )
            r.raise_for_status()
            data = r.json()
        total_ms = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage", {})
        text = data["choices"][0].get("text", "")
        return GenerationResponse(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", len(req.prompt.split())),
            completion_tokens=usage.get("completion_tokens", len(text.split())),
            ttft_ms=total_ms,  # non-streaming: TTFT == total. Use stream() for true TTFT.
            total_ms=total_ms,
        )

    async def stream(self, req: GenerationRequest):
        import httpx

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/completions",
                json={
                    "model": self.model,
                    "prompt": req.prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "stream": True,
                },
            ) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and "[DONE]" not in line:
                        yield line[len("data: ") :]

    async def health(self) -> bool:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/models")
                return r.status_code == 200
        except Exception:
            return False

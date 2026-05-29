"""Naive transformers backend — Layer 0 baseline.

A deliberately unoptimized generation loop (no batching) so the later vLLM
numbers have something to beat. Runs on CPU with a tiny model or on a GPU.
Requires `pip install -r requirements-gpu.txt` (torch + transformers).
"""
from __future__ import annotations

import time

from ..schemas import GenerationRequest, GenerationResponse


class TransformersBackend:
    name = "transformers"
    simulated = False

    def __init__(self, model: str, dtype: str = "fp16") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as e:  # pragma: no cover - GPU phase only
            raise RuntimeError(
                "TransformersBackend needs torch + transformers. "
                "Install with: pip install -r requirements-gpu.txt"
            ) from e

        self.model_name = model
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
        ).to(self._device)
        self.profile = {"model": model, "dtype": dtype, "engine": "transformers", "device": self._device}

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        torch = self._torch
        t0 = time.perf_counter()
        inputs = self._tok(req.prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=req.max_tokens, do_sample=req.temperature > 0
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        text = self._tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return GenerationResponse(
            text=text,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(out.shape[1] - inputs["input_ids"].shape[1]),
            ttft_ms=total_ms,  # naive loop: no streaming, TTFT == total
            total_ms=total_ms,
        )

    async def stream(self, req: GenerationRequest):
        resp = await self.generate(req)
        yield resp.text

    async def health(self) -> bool:
        return True

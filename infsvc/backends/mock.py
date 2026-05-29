"""Synthetic backend — the thing that makes the whole project runnable offline.

It carries a small analytical performance model that reproduces the *qualitative*
story of the project (naive -> vLLM -> quantized -> tuned): no batching means
throughput stays flat while latency balloons with concurrency; continuous
batching makes throughput scale and latency hold; quantization cuts memory at a
small quality cost.

EVERY result it produces is marked synthetic=True. The numbers show the *shape*
of the optimization story; only a real GPU gives defensible magnitudes. Replace
this with VLLMBackend on a GPU and the identical benchmark harness emits real
numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import BenchmarkConfig
from ..cost import cost_per_1k_tokens
from ..schemas import BenchmarkResult, GenerationRequest, GenerationResponse


@dataclass(frozen=True)
class Profile:
    tok_s: float            # single-stream decode tokens/sec
    max_batch: int          # concurrent streams the engine batches
    batch_eff: float        # 0..1: how well throughput scales with batch size
    mem_gb: float
    quality: float          # relative to fp16 baseline
    ttft_base_ms: float
    prefill_ms_per_tok: float
    gpu_util: float         # representative utilization at full batch


# The story, encoded. naive can't batch; vLLM can; awq is lighter; tuned is best.
PROFILES: dict[str, Profile] = {
    "naive": Profile(30, 1, 0.0, 15.0, 1.000, 80, 0.060, 35),
    "vllm": Profile(32, 64, 0.90, 15.0, 1.000, 60, 0.040, 88),
    "awq": Profile(40, 64, 0.90, 6.0, 0.985, 55, 0.035, 85),
    "tuned": Profile(42, 96, 0.94, 7.0, 0.985, 48, 0.030, 92),
}


def _jitter(seed: str) -> float:
    """Deterministic ±, so percentiles look non-synthetic without using random."""
    h = int(__import__("hashlib").md5(seed.encode()).hexdigest(), 16)
    return 0.95 + (h % 1000) / 10000.0  # 0.95 .. 1.05


class MockBackend:
    name = "mock"
    simulated = True

    def __init__(self, profile: str = "vllm") -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown mock profile: {profile}")
        self.profile_name = profile
        self._p = PROFILES[profile]
        self.profile = {
            "model": "mock-sim",
            "dtype": profile,
            "max_batch": self._p.max_batch,
            "gpu_mem_gb": self._p.mem_gb,
        }

    # --- analytical benchmark path (used by the harness; no real timing) ---
    def simulate(self, cfg: BenchmarkConfig) -> BenchmarkResult:
        p = self._p
        c = max(1, cfg.concurrency)
        batch = min(c, p.max_batch)

        # Per-stream decode slows as the batch grows (shared GPU); efficiency
        # controls how gently. naive (batch_eff=0) never batches.
        per_stream = p.tok_s / (1 + (batch - 1) * (1 - p.batch_eff))
        aggregate_tok_s = per_stream * batch

        prefill_ms = p.ttft_base_ms + cfg.input_tokens * p.prefill_ms_per_tok
        decode_ms = (cfg.output_tokens / per_stream) * 1000.0
        service_ms = prefill_ms + decode_ms

        # Requests beyond max_batch queue in "waves".
        waves = math.ceil(c / p.max_batch)
        mean_ms = service_ms * (waves + 1) / 2.0
        ttft_mean = prefill_ms + service_ms * (waves - 1) / 2.0

        j = _jitter(cfg.hash)
        req_per_s = aggregate_tok_s / cfg.output_tokens
        util = min(99.0, p.gpu_util * (0.5 + 0.5 * batch / p.max_batch))
        mem = p.mem_gb + 0.01 * batch  # KV cache grows a little with batch

        return BenchmarkResult(
            config_hash=cfg.hash,
            label=cfg.label,
            backend="mock",
            profile=self.profile_name,
            dtype=cfg.dtype,
            concurrency=c,
            num_requests=cfg.num_requests,
            input_tokens=cfg.input_tokens,
            output_tokens=cfg.output_tokens,
            req_per_s=round(req_per_s, 3),
            output_tok_per_s=round(aggregate_tok_s, 1),
            lat_p50_ms=round(mean_ms * j, 1),
            lat_p95_ms=round(mean_ms * 1.4 * j, 1),
            lat_p99_ms=round(mean_ms * 1.8 * j, 1),
            ttft_p50_ms=round(ttft_mean * j, 1),
            ttft_p99_ms=round(ttft_mean * 1.8 * j, 1),
            gpu_mem_gb=round(mem, 2),
            gpu_util_pct=round(util, 1),
            cost_per_1k_usd=round(cost_per_1k_tokens(cfg.gpu_hourly_usd, aggregate_tok_s), 6),
            quality=p.quality,
            errors=0,
            synthetic=True,
        )

    # --- per-request path (used by the gateway when serving with mock) ---
    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        p = self._p
        prompt_tokens = max(1, len(req.prompt.split()))
        completion = min(req.max_tokens, 256)
        prefill_ms = p.ttft_base_ms + prompt_tokens * p.prefill_ms_per_tok
        total_ms = prefill_ms + (completion / p.tok_s) * 1000.0
        text = " ".join(["tok"] * completion)
        return GenerationResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion,
            ttft_ms=round(prefill_ms, 2),
            total_ms=round(total_ms, 2),
        )

    async def stream(self, req: GenerationRequest):
        resp = await self.generate(req)
        for tok in resp.text.split():
            yield tok + " "

    async def health(self) -> bool:
        return True

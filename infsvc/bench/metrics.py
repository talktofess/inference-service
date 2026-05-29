"""Aggregation for the real-timing benchmark path: per-request samples + total
wall time -> a BenchmarkResult.
"""
from __future__ import annotations

from ..config import BenchmarkConfig
from ..cost import cost_per_1k_tokens
from ..schemas import BenchmarkResult, RequestSample


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def aggregate(cfg: BenchmarkConfig, samples: list[RequestSample], wall_s: float) -> BenchmarkResult:
    ok = [s for s in samples if s.ok]
    errors = len(samples) - len(ok)
    total_tokens = sum(s.completion_tokens for s in ok)
    tok_s = total_tokens / wall_s if wall_s > 0 else 0.0

    lat = [s.total_ms for s in ok]
    ttft = [s.ttft_ms for s in ok]

    return BenchmarkResult(
        config_hash=cfg.hash,
        label=cfg.label,
        backend=cfg.backend,
        profile=cfg.profile,
        dtype=cfg.dtype,
        concurrency=cfg.concurrency,
        num_requests=cfg.num_requests,
        input_tokens=cfg.input_tokens,
        output_tokens=cfg.output_tokens,
        req_per_s=round(len(ok) / wall_s, 3) if wall_s > 0 else 0.0,
        output_tok_per_s=round(tok_s, 1),
        lat_p50_ms=round(percentile(lat, 50), 1),
        lat_p95_ms=round(percentile(lat, 95), 1),
        lat_p99_ms=round(percentile(lat, 99), 1),
        ttft_p50_ms=round(percentile(ttft, 50), 1),
        ttft_p99_ms=round(percentile(ttft, 99), 1),
        gpu_mem_gb=None,        # real GPU mem/util come from the engine's own metrics
        gpu_util_pct=None,
        cost_per_1k_usd=round(cost_per_1k_tokens(cfg.gpu_hourly_usd, tok_s), 6),
        quality=None,
        errors=errors,
        synthetic=False,
    )

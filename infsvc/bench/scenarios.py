"""Canonical offline scenarios for the headline artifacts.

`comparison_grid` -> the four systems at a fixed concurrency (the master table).
`concurrency_sweep` -> one profile across many concurrencies (the curve).

gpu_hourly_usd per profile reflects the GPU each realistically needs: fp16
(naive/vllm) wants ~16GB (pricier), quantized fits ~8GB (cheaper) — which is
exactly why $/1k tok drops on top of the throughput gain.
"""
from __future__ import annotations

from ..config import BenchmarkConfig

_PRICE = {"naive": 1.00, "vllm": 1.00, "awq": 0.60, "tuned": 0.60}


def comparison_grid(concurrency: int = 32) -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(backend="mock", profile=p, dtype=p, concurrency=concurrency,
                        gpu_hourly_usd=_PRICE[p])
        for p in ("naive", "vllm", "awq", "tuned")
    ]


def concurrency_sweep(profiles=("naive", "vllm", "tuned"),
                      levels=(1, 2, 4, 8, 16, 32, 64, 128)) -> list[BenchmarkConfig]:
    cfgs = []
    for p in profiles:
        for c in levels:
            cfgs.append(
                BenchmarkConfig(backend="mock", profile=p, dtype=p, concurrency=c,
                                gpu_hourly_usd=_PRICE[p])
            )
    return cfgs

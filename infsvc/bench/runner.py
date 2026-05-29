"""Benchmark runner.

Two paths, one interface:
  - simulated backends (MockBackend) return an analytical BenchmarkResult — fast,
    deterministic, no real timing.
  - real backends are driven with a concurrency-limited async load and measured
    by wall clock.
"""
from __future__ import annotations

import asyncio
from time import perf_counter

from ..backends.base import get_backend
from ..config import BenchmarkConfig
from ..schemas import BenchmarkResult, RequestSample
from . import metrics
from .workload import build_workload


async def _run_real(cfg: BenchmarkConfig, backend) -> BenchmarkResult:
    workload = build_workload(cfg)
    sem = asyncio.Semaphore(cfg.concurrency)
    samples: list[RequestSample] = []

    async def one(req) -> None:
        async with sem:
            t0 = perf_counter()
            try:
                resp = await backend.generate(req)
                samples.append(
                    RequestSample(
                        ok=True,
                        ttft_ms=resp.ttft_ms,
                        total_ms=(perf_counter() - t0) * 1000.0,
                        completion_tokens=resp.completion_tokens,
                    )
                )
            except Exception:
                samples.append(RequestSample(ok=False))

    wall0 = perf_counter()
    await asyncio.gather(*(one(r) for r in workload))
    wall = perf_counter() - wall0
    return metrics.aggregate(cfg, samples, wall)


async def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    backend = get_backend(cfg)
    if getattr(backend, "simulated", False):
        return backend.simulate(cfg)
    return await _run_real(cfg, backend)


def run(cfg: BenchmarkConfig) -> BenchmarkResult:
    """Synchronous convenience wrapper."""
    return asyncio.run(run_benchmark(cfg))

"""Data contracts shared across backends, gateway, and benchmark harness."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


class GenerationResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float           # time to first token
    total_ms: float


class RequestSample(BaseModel):
    """One request's measured outcome on the real-timing benchmark path."""
    ok: bool
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    completion_tokens: int = 0


class BenchmarkResult(BaseModel):
    # --- identity ---
    config_hash: str
    label: str               # "naive" | "vllm" | "vllm:awq" ...
    backend: str
    profile: str
    dtype: str
    concurrency: int
    num_requests: int
    input_tokens: int
    output_tokens: int

    # --- throughput ---
    req_per_s: float
    output_tok_per_s: float

    # --- latency (ms) ---
    lat_p50_ms: float
    lat_p95_ms: float
    lat_p99_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float

    # --- resource / cost ---
    gpu_mem_gb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    cost_per_1k_usd: float = float("inf")
    quality: Optional[float] = None     # relative to fp16 baseline (1.0)

    # --- bookkeeping ---
    errors: int = 0
    synthetic: bool = False             # True => numbers are illustrative, not measured
    created_at: str = ""

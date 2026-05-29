"""Configuration objects.

`BenchmarkConfig` is frozen and its `.hash` keys every stored benchmark run —
so a row in the master comparison table is always reproducible from the exact
knobs (backend, dtype, concurrency, workload) that produced it. Same pattern as
Project 1's eval Config.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Literal

from pydantic import BaseModel

Backend = Literal["mock", "transformers", "vllm", "tgi"]


class BenchmarkConfig(BaseModel):
    model_config = {"frozen": True}

    backend: Backend = "mock"
    model: str = "sim"
    # When backend == "mock", `profile` selects which system to emulate.
    # For real backends it's informational; `dtype` drives quantization.
    profile: Literal["naive", "vllm", "awq", "tuned"] = "vllm"
    dtype: str = "fp16"

    concurrency: int = 32
    num_requests: int = 256
    input_tokens: int = 512
    output_tokens: int = 128

    gpu_hourly_usd: float = 0.0   # used to derive cost-per-1k-tokens

    @property
    def label(self) -> str:
        return self.profile if self.backend == "mock" else f"{self.backend}:{self.dtype}"

    @property
    def hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


class ServiceConfig(BaseModel):
    """Gateway runtime config, sourced from environment."""
    api_keys: list[str] = ["demo-key"]
    rate_rpm: int = 120
    primary_backend: Backend = "mock"
    primary_profile: str = "vllm"
    fallback_backend: Backend = "mock"
    fallback_profile: str = "awq"
    model: str = "sim"

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        keys = os.getenv("INFSVC_API_KEYS", "demo-key")
        return cls(
            api_keys=[k.strip() for k in keys.split(",") if k.strip()],
            rate_rpm=int(os.getenv("INFSVC_RATE_RPM", "120")),
            primary_backend=os.getenv("INFSVC_PRIMARY_BACKEND", "mock"),  # type: ignore[arg-type]
            primary_profile=os.getenv("INFSVC_PRIMARY_PROFILE", "vllm"),
            fallback_backend=os.getenv("INFSVC_FALLBACK_BACKEND", "mock"),  # type: ignore[arg-type]
            fallback_profile=os.getenv("INFSVC_FALLBACK_PROFILE", "awq"),
            model=os.getenv("INFSVC_MODEL", "sim"),
        )

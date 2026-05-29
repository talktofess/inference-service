"""The InferenceBackend seam + a factory.

Everything else (gateway, benchmark harness, observability) depends only on this
protocol — never on a model. Swap the implementation, keep the system.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..schemas import GenerationRequest, GenerationResponse


@runtime_checkable
class InferenceBackend(Protocol):
    name: str
    profile: dict            # static facts: model, dtype, max_batch, gpu_mem_gb
    simulated: bool          # True => analytical, no real model/timing

    async def generate(self, req: GenerationRequest) -> GenerationResponse: ...
    async def stream(self, req: GenerationRequest) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...


def get_backend(cfg) -> InferenceBackend:
    """Build a backend from any object exposing .backend/.profile/.model/.dtype."""
    backend = getattr(cfg, "backend", None) or getattr(cfg, "primary_backend", "mock")
    profile = getattr(cfg, "profile", None) or getattr(cfg, "primary_profile", "vllm")
    model = getattr(cfg, "model", "sim")
    dtype = getattr(cfg, "dtype", "fp16")

    if backend == "mock":
        from .mock import MockBackend

        return MockBackend(profile)
    if backend == "transformers":
        from .transformers import TransformersBackend

        return TransformersBackend(model, dtype)
    if backend == "vllm":
        from .vllm import VLLMBackend

        return VLLMBackend(model)
    if backend == "tgi":
        from .tgi import TGIBackend

        return TGIBackend(model)
    raise ValueError(f"unknown backend: {backend}")

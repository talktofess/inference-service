"""Workload generation for a benchmark run."""
from __future__ import annotations

from ..config import BenchmarkConfig
from ..schemas import GenerationRequest


def build_workload(cfg: BenchmarkConfig) -> list[GenerationRequest]:
    """A fixed-shape workload: `num_requests` prompts of `input_tokens` length,
    each requesting `output_tokens`. Deterministic so runs are comparable."""
    prompt = " ".join(["word"] * cfg.input_tokens)
    return [
        GenerationRequest(prompt=prompt, max_tokens=cfg.output_tokens, temperature=0.0)
        for _ in range(cfg.num_requests)
    ]

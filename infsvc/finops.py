"""FinOps: turn throughput + GPU price into capacity-planning and cost-comparison
numbers. Pure arithmetic on top of cost.py — fully testable offline; the
throughput inputs come from the benchmark harness (real on a GPU, synthetic on
the mock).
"""
from __future__ import annotations

from .cost import cost_per_1k_tokens, monthly_cost_at_qps


def break_even_qps(tokens_per_sec_capacity: float, tokens_per_req: int,
                   utilization: float = 1.0) -> float:
    """The request rate one replica can sustain before needing to scale out."""
    if tokens_per_req <= 0:
        return 0.0
    return (tokens_per_sec_capacity * utilization) / tokens_per_req


def compare(systems: list[dict], qps: float, tokens_per_req: int,
            utilization: float = 0.7) -> list[dict]:
    """Cost comparison across systems at a target QPS.

    Each system: {"label", "gpu_hourly", "tok_s"}. Returns per-system replicas,
    monthly cost, $/1k tokens, and single-replica break-even QPS — sorted cheapest
    first.
    """
    rows = []
    for s in systems:
        plan = monthly_cost_at_qps(qps, tokens_per_req, s["gpu_hourly"], s["tok_s"], utilization)
        rows.append({
            "label": s["label"],
            "replicas": plan["replicas"],
            "monthly_usd": plan["monthly_usd"],
            "cost_per_1k_usd": round(cost_per_1k_tokens(s["gpu_hourly"], s["tok_s"]), 6),
            "break_even_qps": round(break_even_qps(s["tok_s"], tokens_per_req, utilization), 2),
        })
    rows.sort(key=lambda r: r["monthly_usd"])
    return rows

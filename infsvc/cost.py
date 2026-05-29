"""Cost model — the bridge from raw throughput to $/1k-tokens. Pure arithmetic,
fully testable with no hardware.
"""
from __future__ import annotations


def cost_per_1k_tokens(gpu_hourly_usd: float, tokens_per_sec: float) -> float:
    """USD to generate 1,000 output tokens at a sustained throughput."""
    if tokens_per_sec <= 0 or gpu_hourly_usd <= 0:
        return float("inf") if gpu_hourly_usd > 0 else 0.0
    tokens_per_hour = tokens_per_sec * 3600.0
    return (gpu_hourly_usd / tokens_per_hour) * 1000.0


def monthly_cost_at_qps(
    qps: float,
    tokens_per_req: int,
    gpu_hourly_usd: float,
    tokens_per_sec_capacity: float,
    utilization: float = 1.0,
) -> dict:
    """Estimate monthly cost to serve a target QPS.

    Sizes the number of GPU replicas needed to meet demand at the given
    per-replica throughput capacity and utilization headroom, then prices them
    for a 730-hour month.
    """
    demand_tok_s = qps * tokens_per_req
    effective_capacity = max(tokens_per_sec_capacity * utilization, 1e-9)
    replicas = max(1, _ceil(demand_tok_s / effective_capacity))
    hours_per_month = 730.0
    monthly = replicas * gpu_hourly_usd * hours_per_month
    return {
        "replicas": replicas,
        "demand_tokens_per_sec": round(demand_tok_s, 2),
        "monthly_usd": round(monthly, 2),
        "cost_per_1k_usd": round(cost_per_1k_tokens(gpu_hourly_usd, tokens_per_sec_capacity), 6),
    }


def _ceil(x: float) -> int:
    import math

    return int(math.ceil(x))

"""Prometheus instrumentation. Degrades to no-ops if prometheus_client is absent
so the gateway can still run in a minimal environment.
"""
from __future__ import annotations

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    _ENABLED = True
except ModuleNotFoundError:  # pragma: no cover
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain"

if _ENABLED:
    REQUESTS = Counter("infsvc_requests_total", "Requests", ["backend", "status"])
    TOKENS = Counter("infsvc_tokens_total", "Tokens generated", ["backend", "kind"])
    LATENCY = Histogram("infsvc_request_latency_ms", "Request latency (ms)", ["backend"],
                        buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000))
    COST = Counter("infsvc_cost_usd_total", "Cumulative serving cost (USD)", ["backend"])
    INFLIGHT = Gauge("infsvc_inflight_requests", "In-flight requests")


def record_request(backend: str, status: str, latency_ms: float,
                   prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    if not _ENABLED:
        return
    REQUESTS.labels(backend, status).inc()
    LATENCY.labels(backend).observe(latency_ms)
    TOKENS.labels(backend, "prompt").inc(prompt_tokens)
    TOKENS.labels(backend, "completion").inc(completion_tokens)
    COST.labels(backend).inc(cost_usd)


def inflight_inc() -> None:
    if _ENABLED:
        INFLIGHT.inc()


def inflight_dec() -> None:
    if _ENABLED:
        INFLIGHT.dec()


def metrics_payload() -> tuple[bytes, str]:
    if not _ENABLED:
        return b"# prometheus_client not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST

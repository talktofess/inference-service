"""FastAPI gateway — the single operable front door.

OpenAI-compatible `/v1/completions` so existing clients/load tools just work,
plus auth, per-key rate limiting, primary->fallback routing, per-request cost
accounting, and Prometheus `/metrics`.

Run:  uvicorn infsvc.gateway.app:app --port 8080      (or: infsvc serve)
"""
from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..backends.base import get_backend
from ..config import ServiceConfig
from ..cost import cost_per_1k_tokens
from ..schemas import GenerationRequest
from . import observability as obs
from .auth import ApiKeyAuth, key_id
from .ratelimit import RateLimiter
from .router import Router

app = FastAPI(title="Inference Gateway")

_cfg = ServiceConfig.from_env()
_auth = ApiKeyAuth(_cfg.api_keys)
_limiter = RateLimiter(_cfg.rate_rpm)


def _build_router() -> Router:
    primary = get_backend(_cfg)
    fallback = None
    if _cfg.fallback_backend:
        fb_view = type("V", (), {"backend": _cfg.fallback_backend,
                                 "profile": _cfg.fallback_profile,
                                 "model": _cfg.model, "dtype": "fp16"})()
        fallback = get_backend(fb_view)
    return Router(primary, fallback)


_router = _build_router()
# A rough $/hr for cost accounting on served traffic (mock/demo default).
_GPU_HOURLY = 1.0


class CompletionRequest(BaseModel):
    model: str = "sim"
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "primary": _router.primary.name,
            "fallback": _router.fallback.name if _router.fallback else None}


@app.get("/metrics")
async def metrics() -> Response:
    payload, content_type = obs.metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/v1/completions")
async def completions(
    body: CompletionRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    api_key = ApiKeyAuth.extract(authorization, x_api_key)
    if not _auth.check(api_key):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    kid = key_id(api_key)
    if not _limiter.allow(kid):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    req = GenerationRequest(prompt=body.prompt, max_tokens=body.max_tokens, temperature=body.temperature)
    obs.inflight_inc()
    t0 = perf_counter()
    try:
        resp, served_by = await _router.generate(req)
    except Exception as e:
        obs.record_request("error", "error", (perf_counter() - t0) * 1000.0, 0, 0, 0.0)
        raise HTTPException(status_code=502, detail=f"backend error: {e}")
    finally:
        obs.inflight_dec()

    latency_ms = (perf_counter() - t0) * 1000.0
    tok_s = resp.completion_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
    cost = cost_per_1k_tokens(_GPU_HOURLY, tok_s) * (resp.completion_tokens / 1000.0)
    obs.record_request(served_by, "ok", latency_ms, resp.prompt_tokens, resp.completion_tokens, cost)

    return JSONResponse(
        {
            "model": body.model,
            "choices": [{"text": resp.text, "index": 0, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.prompt_tokens + resp.completion_tokens,
            },
            "served_by": served_by,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost, 8),
        }
    )

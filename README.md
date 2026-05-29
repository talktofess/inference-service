# Optimized Model Inference Service

Serve an open-weights model behind a production-style API and **prove the
optimizations with numbers**. The headline isn't "a model runs" — it's a
before/after benchmark showing measurable gains in throughput, latency, and
cost-per-token across `naive → vLLM → quantized → tuned`.

> Implements the scope in [`../02-model-inference-service.md`](../02-model-inference-service.md).
> **CPU-first scaffold:** the serving engine sits behind an `InferenceBackend`
> interface, so a synthetic `MockBackend` stands in for vLLM and the entire
> system — gateway, benchmark harness, cost model, dashboards — runs offline
> today. Swap one implementation on a GPU and the same harness emits real numbers.

## Architecture

```
load test ─┐
benchmark ─┤─▶ FastAPI gateway ─▶ InferenceBackend ─▶ [engine]
           │     auth · rate limit     │  mock | transformers
           │     routing/fallback      │  vllm | tgi (HTTP)
           │     cost + token metrics  ▼
           ▼                       /metrics ─▶ Prometheus ─▶ Grafana
```

| Layer | Module | Status |
|---|---|---|
| Backend seam + cost model | `infsvc/backends/base.py`, `infsvc/cost.py` | ✅ |
| Synthetic backend (offline) | `infsvc/backends/mock.py` | ✅ |
| Benchmark harness | `infsvc/bench/` (workload, runner, metrics, store, report) | ✅ |
| Gateway | `infsvc/gateway/` (auth, ratelimit, router, observability, app) | ✅ |
| Observability as code | `observability/` (prometheus, grafana, alerts) | ✅ |
| Docker + load tests | `docker/`, `loadtest/` | ✅ |
| Naive baseline (Layer 0) | `infsvc/backends/transformers.py` | ✅ needs GPU deps |
| vLLM / TGI (Layers 1-3) | `infsvc/backends/vllm.py`, `tgi.py` | ✅ needs a running engine |

## Run it offline right now (no GPU, no API key)

```powershell
cd "inference-service"
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements-test.txt        # just pydantic + pytest
python -m infsvc.cli sweep --fresh           # canonical mock benchmarks
python -m infsvc.cli report                  # master comparison table (+ curve w/ matplotlib)
python -m infsvc.cli costcalc --gpu-hourly 1.0 --tok-s 250 --qps 10
pytest -q
```

Example `report` output (synthetic — illustrative shapes, not measured):

```
> SYNTHETIC - illustrative only.
| system | tok/s | p99 latency (ms) | $/1k tok | GPU mem (GB) | quality | util % |
|--------|-------|------------------|----------|--------------|---------|--------|
| naive  |    30 |           129371 |   0.0093 |         15.0 |   1.000 |     35 |
| vllm   |   250 |            28247 |   0.0011 |         15.3 |   1.000 |     66 |
| awq    |   312 |            24156 |   0.0005 |          6.3 |   0.985 |     64 |
| tuned  |   470 |            16045 |   0.0004 |          7.3 |   0.985 |     61 |
```

The story is right (continuous batching lifts throughput and tames tail
latency; quantization cuts memory ~60% at a ~1.5% quality cost; cost/token
falls), but the **magnitudes are synthetic** — every mock report is stamped
`SYNTHETIC`. Real numbers come from the GPU phase below.

## The ops stack (offline)

```bash
docker compose -f docker/docker-compose.yml up      # gateway + Prometheus + Grafana
# drive load so the dashboard populates:
locust -f loadtest/locustfile.py --host http://localhost:8080
# Grafana at http://localhost:3000  ->  "Inference Service" dashboard
```

## The gateway

OpenAI-compatible `/v1/completions` with API-key auth, per-key token-bucket
rate limiting, primary→fallback routing (e.g. a smaller model under load),
per-request cost accounting, and Prometheus `/metrics`.

```bash
infsvc serve                  # uvicorn infsvc.gateway.app:app
curl -H "x-api-key: demo-key" -d '{"prompt":"hi","max_tokens":32}' \
     http://localhost:8080/v1/completions
```

## The GPU phase (real numbers)

On a Linux/WSL2 box with an NVIDIA GPU:

```bash
pip install -r requirements-gpu.txt
# Layer 0 baseline:
infsvc bench --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 1
# Layer 1+: start vLLM, then benchmark it:
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000 &
infsvc bench --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 32 --gpu-hourly 1.0
infsvc report                 # now mixes real (un-stamped) and synthetic rows
```

The same `runner`/`report`/`cost` code paths are used — only the backend
changes — so real results drop straight into the comparison table.

## Design notes

- **`InferenceBackend` is the seam.** Gateway, harness, and metrics never import
  a model; they talk to the interface. `get_backend(cfg)` is the one factory.
- **`BenchmarkConfig.hash` keys every stored run**, so each table row is
  reproducible from the exact (backend, dtype, concurrency, workload) knobs.
- **Honesty rule:** any report containing mock rows is stamped `SYNTHETIC`.
  Faking benchmark numbers would sink this as a portfolio piece.
- **Cost model is pure arithmetic** (`cost.py`) — `$/1k tok = $/hr ÷ tok/hr` —
  and fully unit-tested without hardware.

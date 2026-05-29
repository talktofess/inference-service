"""Command-line entry points.

  infsvc sweep                 # run the canonical offline (mock) benchmark set
  infsvc bench --profile vllm --concurrency 32
  infsvc report                # master comparison table + throughput/latency curve
  infsvc costcalc --gpu-hourly 1.0 --tok-s 250
  infsvc serve                 # launch the gateway (needs fastapi)

Everything except `serve` runs offline with only pydantic installed.
"""
from __future__ import annotations

import argparse
import json
import sys

# Force UTF-8 stdout so markdown tables render on Windows' cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _cmd_sweep(args) -> None:
    from .bench import store
    from .bench.runner import run
    from .bench.scenarios import comparison_grid, concurrency_sweep

    if args.fresh:
        store.clear()
    cfgs = comparison_grid(args.concurrency) + concurrency_sweep()
    print(f"Running {len(cfgs)} mock benchmark configs...")
    for cfg in cfgs:
        store.save(run(cfg))
    print(f"Stored {len(cfgs)} results. Run `infsvc report`.")


def _cmd_bench(args) -> None:
    from .bench import store
    from .bench.runner import run
    from .config import BenchmarkConfig

    cfg = BenchmarkConfig(
        backend=args.backend, profile=args.profile, dtype=args.dtype,
        model=args.model, concurrency=args.concurrency,
        num_requests=args.num_requests, input_tokens=args.input_tokens,
        output_tokens=args.output_tokens, gpu_hourly_usd=args.gpu_hourly,
    )
    result = run(cfg)
    store.save(result)
    print(json.dumps(result.model_dump(), indent=2, default=str))


def _cmd_report(args) -> None:
    from .bench import report, store

    results = store.load_all()
    print(report.master_table(results, concurrency=args.concurrency))
    png = report.write_curve_png(results, profiles=["naive", "vllm", "tuned"])
    if png:
        print(f"\nCurve written to {png}")
    else:
        print("\n(install matplotlib to render the throughput/latency curve)")


def _cmd_costcalc(args) -> None:
    from .cost import cost_per_1k_tokens, monthly_cost_at_qps

    print(f"cost per 1k tokens: ${cost_per_1k_tokens(args.gpu_hourly, args.tok_s):.6f}")
    if args.qps:
        out = monthly_cost_at_qps(args.qps, args.tokens_per_req, args.gpu_hourly, args.tok_s, args.util)
        print(json.dumps(out, indent=2))


def _cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("infsvc.gateway.app:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    p = argparse.ArgumentParser(prog="infsvc")
    sub = p.add_subparsers(required=True)

    ps = sub.add_parser("sweep", help="run canonical offline mock benchmarks")
    ps.add_argument("--concurrency", type=int, default=32)
    ps.add_argument("--fresh", action="store_true", help="clear stored runs first")
    ps.set_defaults(fn=_cmd_sweep)

    pb = sub.add_parser("bench", help="run a single benchmark config")
    pb.add_argument("--backend", default="mock")
    pb.add_argument("--profile", default="vllm", choices=["naive", "vllm", "awq", "tuned"])
    pb.add_argument("--dtype", default="fp16")
    pb.add_argument("--model", default="sim")
    pb.add_argument("--concurrency", type=int, default=32)
    pb.add_argument("--num-requests", type=int, default=256)
    pb.add_argument("--input-tokens", type=int, default=512)
    pb.add_argument("--output-tokens", type=int, default=128)
    pb.add_argument("--gpu-hourly", type=float, default=1.0)
    pb.set_defaults(fn=_cmd_bench)

    pr = sub.add_parser("report", help="render comparison table + curve")
    pr.add_argument("--concurrency", type=int, default=32)
    pr.set_defaults(fn=_cmd_report)

    pc = sub.add_parser("costcalc", help="cost-per-token + monthly estimate")
    pc.add_argument("--gpu-hourly", type=float, required=True)
    pc.add_argument("--tok-s", type=float, required=True, help="throughput capacity (tokens/sec)")
    pc.add_argument("--qps", type=float, default=0.0)
    pc.add_argument("--tokens-per-req", type=int, default=256)
    pc.add_argument("--util", type=float, default=0.7)
    pc.set_defaults(fn=_cmd_costcalc)

    pv = sub.add_parser("serve", help="launch the gateway")
    pv.add_argument("--host", default="0.0.0.0")
    pv.add_argument("--port", type=int, default=8080)
    pv.add_argument("--reload", action="store_true")
    pv.set_defaults(fn=_cmd_serve)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

from infsvc.bench.metrics import aggregate, percentile
from infsvc.config import BenchmarkConfig
from infsvc.schemas import RequestSample


def test_percentile_interpolates():
    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 50) == 30
    assert percentile(vals, 0) == 10
    assert percentile(vals, 100) == 50


def test_aggregate_counts_errors_and_throughput():
    cfg = BenchmarkConfig(backend="vllm", concurrency=2, num_requests=3, output_tokens=100,
                          gpu_hourly_usd=1.0)
    samples = [
        RequestSample(ok=True, ttft_ms=50, total_ms=1000, completion_tokens=100),
        RequestSample(ok=True, ttft_ms=60, total_ms=1200, completion_tokens=100),
        RequestSample(ok=False),
    ]
    res = aggregate(cfg, samples, wall_s=2.0)
    assert res.errors == 1
    assert res.output_tok_per_s == 100.0  # 200 tokens / 2.0s
    assert res.synthetic is False
    assert res.cost_per_1k_usd > 0

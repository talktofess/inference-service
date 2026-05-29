from infsvc.backends.mock import MockBackend
from infsvc.config import BenchmarkConfig


def _sim(profile, concurrency):
    cfg = BenchmarkConfig(backend="mock", profile=profile, dtype=profile,
                          concurrency=concurrency, gpu_hourly_usd=1.0)
    return MockBackend(profile).simulate(cfg)


def test_batching_beats_naive_at_high_concurrency():
    naive = _sim("naive", 32)
    vllm = _sim("vllm", 32)
    # Continuous batching -> far higher throughput and far lower tail latency.
    assert vllm.output_tok_per_s > naive.output_tok_per_s * 5
    assert vllm.lat_p99_ms < naive.lat_p99_ms


def test_quantization_cuts_memory_at_a_quality_cost():
    vllm = _sim("vllm", 32)
    awq = _sim("awq", 32)
    assert awq.gpu_mem_gb < vllm.gpu_mem_gb * 0.6   # ~60% memory cut
    assert awq.quality < vllm.quality                # honest: small quality drop


def test_results_are_marked_synthetic():
    assert _sim("tuned", 16).synthetic is True


def test_naive_throughput_is_flat_across_concurrency():
    # No batching: aggregate throughput barely moves as concurrency rises.
    assert abs(_sim("naive", 1).output_tok_per_s - _sim("naive", 64).output_tok_per_s) < 1.0

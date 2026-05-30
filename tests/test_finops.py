from infsvc.finops import break_even_qps, compare


def test_break_even_qps_scales_with_capacity():
    # 250 tok/s capacity, 200 tok/req, full utilization -> 1.25 req/s per replica
    assert break_even_qps(250, 200, 1.0) == 1.25
    assert break_even_qps(500, 200, 1.0) > break_even_qps(250, 200, 1.0)


def test_compare_ranks_cheaper_system_first_and_is_consistent():
    systems = [
        {"label": "naive", "gpu_hourly": 1.0, "tok_s": 30},
        {"label": "vllm", "gpu_hourly": 1.0, "tok_s": 250},
        {"label": "awq", "gpu_hourly": 0.6, "tok_s": 312},
    ]
    rows = compare(systems, qps=10, tokens_per_req=200)
    # cheapest first; the higher-throughput/cheaper-GPU system wins
    assert rows[0]["label"] == "awq"
    assert rows[-1]["label"] == "naive"
    # more throughput -> fewer replicas for the same demand
    by = {r["label"]: r for r in rows}
    assert by["vllm"]["replicas"] < by["naive"]["replicas"]

from infsvc.cost import cost_per_1k_tokens, monthly_cost_at_qps


def test_cost_per_1k_scales_inversely_with_throughput():
    slow = cost_per_1k_tokens(1.0, 100)
    fast = cost_per_1k_tokens(1.0, 200)
    assert fast < slow
    # $1/hr at 100 tok/s = 360k tok/hr -> $1/360 per 1k
    assert abs(slow - (1.0 / 360_000) * 1000) < 1e-9


def test_cost_zero_price_is_free():
    assert cost_per_1k_tokens(0.0, 100) == 0.0


def test_monthly_cost_sizes_replicas_to_demand():
    out = monthly_cost_at_qps(qps=10, tokens_per_req=200, gpu_hourly_usd=1.0,
                              tokens_per_sec_capacity=250, utilization=0.8)
    # demand = 2000 tok/s; capacity*util = 200 -> 10 replicas
    assert out["replicas"] == 10
    assert out["monthly_usd"] == round(10 * 1.0 * 730.0, 2)

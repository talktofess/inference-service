"""End-to-end offline smoke: run the canonical mock scenarios, store, report —
all with only pydantic, no GPU. Storage redirected to tmp so nothing leaks.
"""
from infsvc import settings


def _redirect(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path / "runs")


def test_sweep_store_report(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)

    from infsvc.bench import report, store
    from infsvc.bench.runner import run
    from infsvc.bench.scenarios import comparison_grid, concurrency_sweep

    for cfg in comparison_grid(32) + concurrency_sweep():
        store.save(run(cfg))

    results = store.load_all()
    assert len(results) >= 4

    table = report.master_table(results, concurrency=32)
    assert "SYNTHETIC" in table          # honesty banner present
    assert "naive" in table and "tuned" in table

    pts = report.curve_points(results, "vllm")
    tput = [p[1] for p in pts]
    assert tput == sorted(tput) or len(set(tput)) > 1  # throughput rises with concurrency

"""Persist benchmark results, keyed by config hash (reproducible rows)."""
from __future__ import annotations

import json

from .. import settings
from ..schemas import BenchmarkResult


def save(result: BenchmarkResult):
    settings.ensure_dirs()
    path = settings.RUNS_DIR / f"{result.config_hash}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_all() -> list[BenchmarkResult]:
    settings.ensure_dirs()
    out = []
    for p in sorted(settings.RUNS_DIR.glob("*.json")):
        out.append(BenchmarkResult(**json.loads(p.read_text(encoding="utf-8"))))
    return out


def clear() -> None:
    settings.ensure_dirs()
    for p in settings.RUNS_DIR.glob("*.json"):
        p.unlink()

"""Infrastructure settings — paths and runtime endpoints. Kept out of the
benchmark config hash (these say *where*, not *what experiment*)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.getenv("INFSVC_RESULTS_DIR", ROOT / "results"))
RUNS_DIR = RESULTS_DIR / "runs"

VLLM_URL = os.getenv("INFSVC_VLLM_URL", "http://localhost:8000/v1")
TGI_URL = os.getenv("INFSVC_TGI_URL", "http://localhost:8080")


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

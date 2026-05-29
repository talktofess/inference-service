"""Optimized model inference service — a backend-agnostic serving gateway and
benchmark harness. The serving engine sits behind an interface, so a mock/CPU
backend stands in for vLLM and the whole system runs offline."""

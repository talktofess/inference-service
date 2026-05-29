"""Locust load test against the gateway.

  locust -f loadtest/locustfile.py --host http://localhost:8080

Drives /v1/completions with auth so the gateway's metrics (and the Grafana
dashboard) populate. Use the web UI to ramp users and watch the throughput vs.
latency tradeoff in real time.
"""
from locust import HttpUser, between, task

API_KEY = "demo-key"


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def complete(self):
        self.client.post(
            "/v1/completions",
            headers={"x-api-key": API_KEY},
            json={"model": "sim", "prompt": "Summarize the theory of relativity.", "max_tokens": 128},
        )

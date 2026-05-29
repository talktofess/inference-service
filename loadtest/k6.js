// k6 load test against the gateway.
//   k6 run loadtest/k6.js
// Ramps concurrency in stages to trace the throughput-vs-latency curve.
import http from "k6/http";
import { check } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 8 },
    { duration: "30s", target: 32 },
    { duration: "30s", target: 64 },
    { duration: "30s", target: 0 },
  ],
};

const API_KEY = __ENV.API_KEY || "demo-key";

export default function () {
  const res = http.post(
    "http://localhost:8080/v1/completions",
    JSON.stringify({ model: "sim", prompt: "Explain paged attention.", max_tokens: 128 }),
    { headers: { "Content-Type": "application/json", "x-api-key": API_KEY } }
  );
  check(res, { "status 200": (r) => r.status === 200 });
}

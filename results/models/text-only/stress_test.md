# Stress test of the text-only model: request latency and throughput

Setup: the Docker image (4 uvicorn workers, 1 encoder thread each) on Docker Desktop on a 10-core Apple M5 MacBook Air with 24 GB (Docker VM: 10 CPUs, 7.7 GB); Locust on the same laptop, through Docker's port forwarding. Each level ran for 30 s after every user had started; Locust measured on the client.

| Requests in flight | Requests | Failures | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2031 | 0 | 69.9 | 16 | 17 | 18 |
| 2 | 3917 | 0 | 134.9 | 17 | 18 | 19 |
| 4 | 6081 | 0 | 209.5 | 22 | 24 | 25 |
| 8 | 7520 | 0 | 259.1 | 31 | 42 | 45 |
| 16 | 7651 | 0 | 263.6 | 59 | 88 | 100 |
| 32 | 7436 | 0 | 256.2 | 120 | 190 | 210 |
| 64 | 7206 | 0 | 248.3 | 260 | 350 | 400 |

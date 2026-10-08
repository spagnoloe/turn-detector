# Stress test: request latency and throughput

Setup: the Docker image (4 uvicorn workers, 1 encoder thread each) on Docker Desktop on a 10-core Apple M5 MacBook Air with 24 GB (Docker VM: 10 CPUs, 7.7 GB); Locust on the same laptop, through Docker's port forwarding. Each level ran for 30 s after every user had started; Locust measured on the client.

| Requests in flight | Requests | Failures | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2028 | 0 | 69.8 | 17 | 18 | 22 |
| 2 | 4031 | 0 | 138.8 | 17 | 18 | 19 |
| 4 | 6228 | 0 | 214.6 | 22 | 23 | 25 |
| 8 | 7824 | 0 | 269.6 | 32 | 39 | 42 |
| 16 | 7959 | 0 | 274.2 | 56 | 90 | 100 |
| 32 | 7905 | 0 | 272.3 | 120 | 160 | 190 |
| 64 | 7440 | 0 | 256.3 | 250 | 350 | 400 |

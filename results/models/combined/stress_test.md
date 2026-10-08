# Stress test of the combined model: request latency and throughput

Setup: the Docker image (4 uvicorn workers, 1 encoder thread each) on Docker Desktop on a 10-core Apple M5 MacBook Air with 24 GB (Docker VM: 10 CPUs, 7.7 GB); Locust on the same laptop, through Docker's port forwarding. Each level ran for 30 s after every user had started; Locust measured on the client.

| Requests in flight | Requests | Failures | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 198 | 0 | 7.1 | 140 | 140 | 150 |
| 2 | 346 | 0 | 11.9 | 170 | 180 | 180 |
| 4 | 464 | 0 | 16 | 250 | 260 | 270 |
| 8 | 581 | 0 | 20 | 400 | 430 | 440 |
| 16 | 540 | 0 | 18.6 | 870 | 1100 | 1200 |
| 32 | 513 | 0 | 17.7 | 1800 | 2300 | 2500 |
| 64 | 487 | 0 | 16.8 | 3600 | 4600 | 5000 |

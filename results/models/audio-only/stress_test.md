# Stress test of the audio-only model: request latency and throughput

Setup: the Docker image (4 uvicorn workers, 1 encoder thread each) on Docker Desktop on a 10-core Apple M5 MacBook Air with 24 GB (Docker VM: 10 CPUs, 7.7 GB); Locust on the same laptop, through Docker's port forwarding. Each level ran for 30 s after every user had started; Locust measured on the client.

| Requests in flight | Requests | Failures | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 210 | 0 | 7.8 | 130 | 130 | 140 |
| 2 | 416 | 0 | 14.3 | 140 | 140 | 150 |
| 4 | 640 | 0 | 22.1 | 180 | 190 | 190 |
| 8 | 889 | 0 | 30.6 | 260 | 270 | 280 |
| 16 | 835 | 0 | 28.7 | 570 | 720 | 780 |
| 32 | 709 | 0 | 24.4 | 1300 | 1800 | 2000 |
| 64 | 579 | 0 | 20 | 3200 | 4000 | 4200 |

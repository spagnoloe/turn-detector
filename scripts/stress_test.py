"""Stress-test the running prediction API at several concurrency levels.

Requests are real: the payloads are what the orchestrator sends while streaming a held-out
conversation (`turn_detector.streaming`): the last 1 s of 16 kHz audio, the transcript so far, the
other speaker's previous turn and the silence duration, sampled across both speakers' pauses.
At each level, Locust runs that many users, each sending requests back to back, so the level is
the number of requests in flight. Statistics are reset once every user is running. It writes
results/serving/stress_test.csv and stress_test.md: request latency p50/p95/p99 and throughput
per level, as Locust measures them on the client.

    docker run --rm -p 8000:8000 turn-detector    # or: uv run uvicorn turn_detector.serving:app
    uv run python scripts/stress_test.py --setup "Docker, 4 CPUs, 4 workers"
"""

import argparse
import csv
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from turn_detector.data import REPO_ROOT, load_audio
from turn_detector.evaluation import load_conversations
from turn_detector.split import load_split
from turn_detector.streaming import pcm_16k, stream

RESULTS_DIR = REPO_ROOT / "results" / "serving"
LOCUSTFILE = Path(__file__).with_name("stress") / "locustfile.py"
N_PAYLOADS = 200
COLUMNS = ["concurrency", "requests", "failures", "throughput_rps", "p50_ms", "p95_ms", "p99_ms"]


def payloads(conversation_id: str, n: int) -> list[dict[str, Any]]:
    """`n` of the requests streaming this conversation sends (all of them if fewer), sampled at random."""
    [conversation] = load_conversations([conversation_id])
    sent = []

    def record(payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        # Never confident, so every pause is streamed up to the trained model's backstop.
        return {"p_eot": 0.0, "threshold": 1.0, "backstop_ms": 1150}

    for side in conversation.sides:
        stream(side, record, pcm_16k(load_audio(conversation_id, side.speaker)))
    return random.Random(0).sample(sent, min(n, len(sent)))


def run_level(url: str, concurrency: int, duration_s: int, payload_path: Path, out_dir: Path) -> dict[str, Any]:
    prefix = out_dir / f"c{concurrency}"
    subprocess.run(
        ["locust", "-f", str(LOCUSTFILE), "--headless", "--host", url, "--users", str(concurrency),
         "--spawn-rate", str(concurrency), "--run-time", f"{duration_s}s", "--reset-stats", "--only-summary",
         "--csv", str(prefix)],
        env={**os.environ, "STRESS_PAYLOADS": str(payload_path)},
        check=True,
        capture_output=True,
    )  # fmt: skip
    with open(f"{prefix}_stats.csv") as file:
        total = next(row for row in csv.DictReader(file) if row["Name"] == "Aggregated")
    return {
        "concurrency": concurrency,
        "requests": int(total["Request Count"]),
        "failures": int(total["Failure Count"]),
        "throughput_rps": round(float(total["Requests/s"]), 1),
        "p50_ms": float(total["50%"]),
        "p95_ms": float(total["95%"]),
        "p99_ms": float(total["99%"]),
    }


def markdown(rows: list[dict[str, Any]], setup: str, duration_s: int) -> str:
    lines = [
        "# Stress test: request latency and throughput",
        "",
        f"Setup: {setup}. Each level ran for {duration_s} s after every user had started; Locust measured on the client.",
        "",
        "| Requests in flight | Requests | Failures | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[column]:g}" for column in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--duration", type=int, default=30, help="seconds per level")
    parser.add_argument("--conversation", default=load_split().held_out[0], help="the held-out conversation to sample")
    parser.add_argument("--setup", required=True, help="what was tested, for the write-up (machine, CPUs, workers)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payloads.json"
        payload_path.write_text(json.dumps(payloads(args.conversation, N_PAYLOADS)))
        for concurrency in args.levels:
            rows.append(run_level(args.url, concurrency, args.duration, payload_path, Path(tmp)))
            print(rows[-1])

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "stress_test.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "stress_test.md").write_text(markdown(rows, args.setup, args.duration))
    print(markdown(rows, args.setup, args.duration))


if __name__ == "__main__":
    main()

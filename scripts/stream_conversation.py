"""Stream a held-out conversation through the running prediction API and print the firings.

Each speaker in turn is the user. Every 50 ms while they are silent, the script sends what the
orchestrator would: the last 1 s of their audio (16 kHz PCM), the transcript so far, the other
speaker's previous turn and the silence duration; it applies the firing rule to each response
(`turn_detector.streaming`). Requests with audio and text are answered by the combined model;
with `--no-text` they carry audio only and the audio-only model answers, and with `--no-audio`
text only and the text-only model answers. It prints every firing with the pause it ended and
the request latency the client saw. The held-out conversations are scored only once, at the end,
so no gold events or scores are shown here.

    uv run uvicorn turn_detector.serving:app        # or the Docker container
    uv run python scripts/stream_conversation.py    # the first held-out conversation
    uv run python scripts/stream_conversation.py --conversation 31 --url http://localhost:8000
"""

import argparse
import time

import httpx2
import numpy as np

from turn_detector.data import load_audio
from turn_detector.evaluation import load_conversations
from turn_detector.split import load_split
from turn_detector.streaming import pcm_16k, stream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    held_out = load_split().held_out
    parser.add_argument("--conversation", default=held_out[0], choices=held_out, help="a held-out conversation id")
    parser.add_argument("--url", default="http://localhost:8000")
    sent = parser.add_mutually_exclusive_group()
    sent.add_argument("--no-audio", action="store_true", help="send text only")
    sent.add_argument("--no-text", action="store_true", help="send audio only")
    args = parser.parse_args()

    [conversation] = load_conversations([args.conversation])
    request_latencies_ms, answered_by = [], set()
    with httpx2.Client(base_url=args.url, timeout=5.0) as client:

        def predict(payload: dict) -> dict:
            start = time.perf_counter()
            response = client.post("/predict", json=payload).raise_for_status()
            request_latencies_ms.append((time.perf_counter() - start) * 1000)
            answered_by.add(response.json()["model"])
            return response.json()

        models = client.get("/health").raise_for_status().json()["models"]
        print(f"serving {', '.join(models)} at {args.url}")
        print(f"conversation {args.conversation} ({conversation.info.conversation_type}, {conversation.duration_s:.0f} s)")
        for side in conversation.sides:
            audio = None if args.no_audio else pcm_16k(load_audio(args.conversation, side.speaker))
            firings = stream(side, predict, audio, text=not args.no_text)
            print(f"\nspeaker {side.speaker} as the user: {len(firings)} firings")
            for firing in firings:
                print(
                    f"  {firing.time_s:8.2f} s  {(firing.time_s - firing.pause_start_s) * 1000:5.0f} ms into the pause"
                    f"  {firing.reason:9}  p_eot {firing.p_eot:.2f}"
                )

    p50, p95, p99 = np.percentile(request_latencies_ms, [50, 95, 99])
    print(f"\nanswered by {', '.join(sorted(answered_by))}")
    print(f"{len(request_latencies_ms)} requests, request latency (client-side) p50 {p50:.1f} / p95 {p95:.1f} / p99 {p99:.1f} ms")


if __name__ == "__main__":
    main()

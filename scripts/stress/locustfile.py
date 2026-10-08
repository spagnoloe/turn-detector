"""Locust users for the stress test: each sends `/predict` requests back to back (a closed loop),
so the number of users is the number of requests in flight. Payloads are read from the JSON file
named by STRESS_PAYLOADS, written by scripts/stress_test.py."""

import json
import os
import random
from pathlib import Path

from locust import FastHttpUser, constant, task

PAYLOADS = json.loads(Path(os.environ["STRESS_PAYLOADS"]).read_text())


class Orchestrator(FastHttpUser):
    wait_time = constant(0)

    @task
    def predict(self) -> None:
        self.client.post("/predict", json=random.choice(PAYLOADS))

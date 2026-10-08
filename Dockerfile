# The prediction API on CPU, with its dependencies, the text-only and audio-only heads and their
# encoders' weights baked in, so it needs no network access. Train the models first (they write
# artifacts/text-only/ and artifacts/audio-only/):
#
#   uv run python scripts/evaluate.py text-only
#   uv run python scripts/evaluate.py audio-only
#   docker build -t turn-detector .
#   docker run --rm -p 8000:8000 turn-detector

FROM python:3.14.8-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
# git fetches the TurnBench dependency.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never HF_HOME=/app/huggingface
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src src
RUN uv sync --locked --no-dev
COPY artifacts/text-only/text-only.json artifacts/text-only/
COPY artifacts/audio-only/audio-only.json artifacts/audio-only/
# Download the encoders the saved heads were trained on, and check the models load and predict.
RUN .venv/bin/python -c "import numpy as np; \
from turn_detector.models.text_only import TextContext, load_text_only; \
from turn_detector.models.audio_only import load_audio_only; \
print(load_text_only().classifier.p_eot([TextContext('Where to?', 'Barcelona.')])); \
print(load_audio_only().classifier.p_eot(np.zeros((1, 16000), np.float32), np.zeros(1)))"

FROM python:3.14.8-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app /app /app
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/huggingface \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=4 \
    OMP_NUM_THREADS=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=2s --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=1)"]
# uvicorn runs WEB_CONCURRENCY worker processes, each with both models loaded and OMP_NUM_THREADS
# threads for the encoders.
CMD ["uvicorn", "turn_detector.serving:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

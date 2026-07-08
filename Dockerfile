FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY app ./app
COPY frontend ./frontend

# The index (data/) is large and built once — mount it at runtime, or build inside
# the container:  docker run <img> uv run python -m app.ingest.build_index
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

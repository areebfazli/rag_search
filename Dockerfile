# Digest-pinned so builds are reproducible and a tag hijack can't swap the base.
# Refresh deliberately: docker pull python:3.12-slim && docker images --digests
FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# uv for fast, reproducible installs (pinned — never :latest in a build)
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

# Create the unprivileged user BEFORE installing, so the venv is written with the
# right ownership already. A `chown -R /app` after `uv sync` rewrites every file in
# the venv, and Docker stores that as a second full copy of it — doubling the image.
RUN useradd --create-home --uid 10001 appuser && chown appuser /app
USER appuser

# Exact lockfile, no glob: a missing uv.lock must fail the build rather than silently
# re-resolving to different versions. --locked fails if the lock is stale.
COPY --chown=appuser pyproject.toml uv.lock ./
RUN uv sync --no-dev --locked --no-cache

COPY --chown=appuser app ./app
COPY --chown=appuser frontend ./frontend

# The index (data/) is large and built once — mount it at runtime, or build inside
# the container:  docker run <img> .venv/bin/python -m app.ingest.build_index
# Pass the LLM key at runtime:  docker run -e SSR_LLM_API_KEY=... ...
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

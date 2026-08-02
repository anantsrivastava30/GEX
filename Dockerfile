# Backend container for the GEX FastAPI service.
#
# Works unchanged on Hugging Face Spaces (Docker SDK), Google Cloud Run,
# Fly, or any Docker host. The host injects the listen port via $PORT;
# Hugging Face Spaces routes to 7860 by default (see deploy/huggingface).
#
# The shared quant_analysis library hard-imports the scientific/plotting
# stack, so we install the full root requirements plus the backend extras.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOME=/app

WORKDIR /app

# Install dependencies first so code changes don't bust the layer cache.
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt

# Application code (see .dockerignore for what is excluded).
COPY quant_analysis ./quant_analysis
COPY backend ./backend
COPY config.yaml ./config.yaml

# Snapshot history is committed under data/snapshots by the daily cron.
# Copy it if present; otherwise the API degrades gracefully to live-only.
COPY data ./data

# Hugging Face Spaces run as a non-root user; make /app writable by it.
RUN chmod -R a+rwX /app
EXPOSE 7860

# Shell form so ${PORT} expands at runtime (HF sets 7860, Cloud Run sets $PORT).
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}

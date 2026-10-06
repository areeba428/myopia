# Full myopia screening app (image + clinical onset branches).
# No PyTorch: inference runs on ONNX Runtime. ~400 MB image.
FROM python:3.12-slim

# Faster, cleaner Python in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /app

# Install deps first for better layer caching
COPY requirements-full.txt .
# PyPI reads time out on a slow link (default timeout is 15s). Retry the install.
RUN set -eu; \
    n=0; \
    until pip install --no-cache-dir --timeout 300 --retries 10 -r requirements-full.txt; do \
      n=$((n+1)); \
      if [ "$n" -ge 3 ]; then exit 1; fi; \
      echo "pip install failed, retry ${n}/3"; \
      sleep 15; \
    done

# App code, model artifacts, original pipelines, frontend
COPY app/ ./app/
COPY code/ ./code/

EXPOSE 8000

# Honor the platform-provided $PORT (Render, HF Spaces) and default to 8000 (Fly).
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

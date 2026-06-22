# Full myopia screening app (image + clinical onset branches).
# No PyTorch: inference runs on ONNX Runtime. ~400 MB image.
FROM python:3.12-slim

# Faster, cleaner Python in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# App code, model artifacts, original pipelines, frontend
COPY app/ ./app/
COPY code/ ./code/

EXPOSE 8000

# Honor the platform-provided $PORT (Render, HF Spaces) and default to 8000 (Fly).
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

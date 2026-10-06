"""API checks for the Jenkins Testing stage.

These do not require the ONNX weights. /api/health and /api/info stay up when
the model artifacts are absent; image scoring then reports the model as unloaded.
"""
from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_info():
    response = client.get("/api/info")
    assert response.status_code == 200
    assert "loaded" in response.json()


def test_frontend_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_rejects_non_image():
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_empty_image():
    response = client.post(
        "/api/predict",
        files={"file": ("fundus.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_predict_missing_model_or_scores():
    """A real PNG either scores (weights present) or returns 503 (weights absent)."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(buf, format="PNG")
    response = client.post(
        "/api/predict",
        files={"file": ("fundus.png", buf.getvalue(), "image/png")},
        data={"heatmap": "false"},
    )
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert "image_risk_score" in body
        assert body["filename"] == "fundus.png"


def test_clinical_status():
    response = client.get("/api/clinical/status")
    assert response.status_code == 200
    body = response.json()
    assert "available" in body or "ready" in body

"""
=============================================================================
 FASTAPI BACKEND  -  My Study myopia screening service
=============================================================================
Serves the genuine IMAGE branch (ResNet-50 + linear-eval head, test AUC 0.988)
as a REST API and hosts the browser frontend.

Endpoints
    GET  /                 -> the frontend (index.html)
    GET  /api/health       -> liveness + whether the model is loaded
    GET  /api/info         -> model card (backbone, protocol, eval metrics)
    POST /api/predict      -> multipart image upload -> myopia risk score

Run:
    uvicorn app.api:app --host 127.0.0.1 --port 8000
    (or)  python app/api.py
=============================================================================
"""
from __future__ import annotations

import os

import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import inference

# The clinical Cox branch needs lifelines/scipy/pandas (~190 MB) — too heavy for a
# minimal serverless bundle. Import it optionally so an image-only deployment
# (e.g. Vercel) still boots; the endpoints then report it as unavailable.
try:
    from . import clinical_service
    CLINICAL_AVAILABLE = True
except Exception as _e:  # missing lifelines/scipy/pandas in a slim deploy
    clinical_service = None
    CLINICAL_AVAILABLE = False
    _CLINICAL_IMPORT_ERROR = str(_e)

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "frontend")

app = FastAPI(
    title="My Study - Myopia Screening API",
    description="Image branch: myopia risk from a retinal fundus photograph.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

ALLOWED = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
MAX_BYTES = 25 * 1024 * 1024  # 25 MB


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": inference.is_ready()}


@app.get("/api/info")
def info():
    return inference.model_info()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), heatmap: bool = Form(True)):
    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{file.content_type}'. Upload a JPEG/PNG fundus image.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 25 MB.")
    try:
        result = inference.predict_bytes(data, heatmap=heatmap)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # malformed image etc.
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
    result["filename"] = file.filename
    return JSONResponse(result)


# ---- clinical onset branch (Cox survival; data collection ongoing) ---------
class OnsetRequest(BaseModel):
    al_cr_ratio: float | None = None
    choroidal_thickness: float | None = None
    choroidal_vascularity_index: float | None = None
    choriocap_flow_deficits: float | None = None
    image_risk_score: float | None = None


def _require_clinical():
    if not CLINICAL_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Clinical onset branch is not enabled in this deployment "
                   "(needs lifelines/scipy/pandas). Run locally or on a container host.")


@app.get("/api/clinical/status")
def clinical_status():
    if not CLINICAL_AVAILABLE:
        return {"ready": False, "available": False,
                "message": "Clinical onset branch not enabled in this deployment "
                           "(image branch only). Runs locally / on a container host."}
    return {"available": True, **clinical_service.status()}


@app.post("/api/clinical/fit")
async def clinical_fit(file: UploadFile = File(...)):
    _require_clinical()
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a cohort .csv file.")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        report = clinical_service.fit_from_csv(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fit cohort: {e}")
    finally:
        os.unlink(tmp_path)
    return {"fitted": True, **report}


@app.post("/api/clinical/predict")
def clinical_predict(req: OnsetRequest):
    _require_clinical()
    try:
        return clinical_service.predict_one(req.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


# ---- frontend (mounted last so it doesn't shadow /api/*) -------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def _warm():
    # Best-effort warm load so the first request isn't slow; ignore if not exported yet.
    try:
        inference.load_model()
        print("[startup] model loaded:", inference.model_info())
    except Exception as e:
        print(f"[startup] model not loaded yet ({e}). "
              "Run app/export_model.py to create the artifact.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)

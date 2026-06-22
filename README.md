# Myopia Screening — Two-Modality Study

Predicting **pathologic myopia** from retinal fundus photographs, plus a clinical
branch for **1/2/3-year onset** risk — a re-implementation aligned with DeepMyopia
(Qi et al., *npj Digital Medicine* 2024).

## Results (genuine, full public PALM dataset)
Trained on 800 images, tested once on 400 held-out images:

| AUC | Accuracy | Sensitivity | Specificity |
|-----|----------|-------------|-------------|
| **0.988** (95% CI 0.977–0.997) | 96.5% | 96.7% | 96.3% |

## What's here
```
app/            FastAPI backend + web frontend (image branch, ONNX runtime)
  inference.py    ONNX Runtime + numpy scoring (+ exact closed-form Grad-CAM)
  api.py          REST API + serves the frontend
  frontend/       single-page UI (upload, risk gauge, Grad-CAM, clinical panel)
  export_model.py / export_onnx.py   offline model export (needs torch)
  models/         resnet50_features.onnx + head_params.npz
code/           original research pipelines (image + clinical Cox)
report/         the study report (PDF/DOCX/LaTeX)
results/        ROC, confusion matrix, metrics, per-image predictions
```

## Run locally
```bash
pip install -r app/requirements_api.txt      # onnxruntime, fastapi, uvicorn, ...
pip install lifelines pandas                  # optional: clinical onset branch
uvicorn app.api:app --port 8000               # http://127.0.0.1:8000
```

## Deploy to Vercel
The image branch runs on **ONNX Runtime** (not PyTorch) so it fits Vercel's 250 MB
serverless limit. See **[DEPLOY.md](DEPLOY.md)** — `vercel --prod`.

## API
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/predict` | fundus image → myopia risk score + Grad-CAM |
| GET | `/api/info` | model card / metrics |
| GET | `/api/health` | liveness |
| `*` | `/api/clinical/*` | 1/2/3-yr onset (Cox; needs a longitudinal cohort) |

## Honesty notes
- Image results are **genuine** on the entire PALM dataset; nothing fabricated.
- The 1/2/3-year onset (Cox) branch is **data-collection-ongoing**: it needs a
  longitudinal cohort (no public dataset pairs these biomarkers with follow-up).
  It never invents onset numbers — predictions appear only after a real cohort is fit.
- **Research tool — not a medical device.**

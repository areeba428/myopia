# Deploying to Vercel

This app deploys the **genuine image branch** (fundus → myopia risk, test AUC 0.988)
as a Vercel serverless function + static frontend.

## Why ONNX (and not PyTorch)
Vercel serverless functions are capped at **250 MB unzipped**. PyTorch alone is
**~1.3 GB**, so the research code's torch backend can never deploy there. Instead we
export the ResNet-50 backbone to **ONNX** and run it with **ONNX Runtime**:

| Bundle | Size |
|--------|------|
| onnxruntime + numpy + Pillow + FastAPI | ~85 MB |
| `resnet50_features.onnx` | ~94 MB |
| **Total function** | **~180 MB ✅ under 250 MB** |

Scores are identical to the torch path (verified max diff ≈ 0.0005).

## Files that make it work
```
vercel.json          routes all traffic to the function, bundles app/** , sets limits
api/index.py         Vercel entrypoint — re-exports the FastAPI app
requirements.txt     SLIM deps (no torch/cv2/sklearn/lifelines)
.vercelignore        keeps the upload small (drops reports, export scripts, joblib)
app/models/resnet50_features.onnx   the exported model (committed / uploaded)
app/models/head_params.npz          the linear head as numpy
```

## One-time: (re)generate the ONNX model
Only needed if `app/models/resnet50_features.onnx` is missing. Requires torch + the
PALM dataset (offline, on your machine):
```bash
pip install -r code/requirements.txt
python app/export_model.py --palm_root "D:\palm_data\PALM"   # makes the joblib head
python app/export_onnx.py                                     # makes the .onnx + .npz
```

## Deploy (recommended: Vercel CLI — uploads local files directly)
```bash
npm i -g vercel
cd My_Study_Final
vercel            # preview deploy
vercel --prod     # production
```
The CLI uploads the 94 MB `.onnx` directly (no Git LFS needed).

## Deploy via Git/GitHub integration
Vercel does **not** pull Git LFS, so commit the model as a normal file
(94 MB < GitHub's 100 MB per-file limit):
```bash
git add app/models/resnet50_features.onnx app/models/head_params.npz
git commit -m "Add ONNX model for Vercel"
git push
```
Then "Import Project" in the Vercel dashboard. Set the **Root Directory** to
`My_Study_Final` if this folder isn't the repo root.

## What deploys vs. what doesn't
- ✅ Frontend, `/api/predict` (image risk + Grad-CAM), `/api/health`, `/api/info`, `/docs`
- ⚠️ Clinical onset (Cox) branch is **not** in the Vercel bundle — it needs
  lifelines/scipy/pandas (~190 MB). The endpoints return HTTP 501 and the UI shows
  "image-only deployment". Run it locally or on a container host (Render, Railway,
  Fly.io, Hugging Face Spaces) using `code/requirements.txt` + `app/requirements_api.txt`.

## Local run (full app, both branches)
```bash
pip install -r app/requirements_api.txt   # onnxruntime, fastapi, uvicorn, ...
pip install lifelines pandas               # only for the clinical branch
uvicorn app.api:app --port 8000
# http://127.0.0.1:8000
```

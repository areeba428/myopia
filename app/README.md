# My Study — Myopia Screening App (FastAPI + Web Frontend)

A web app around the **genuine image branch** of the two-modality My Study system:
upload a retinal fundus photograph → get a myopia risk score from the
ImageNet ResNet-50 backbone + the linear-evaluation logistic-regression head
(held-out PALM test AUC = 0.988).

## Layout
```
app/
  export_model.py      # fit + SAVE the classifier head from PALM features (run once)
  inference.py         # load backbone + saved head, score one image
  api.py               # FastAPI server + serves the frontend
  frontend/index.html  # single-page UI (upload, gauge, metrics)
  models/              # myopia_classifier.joblib  (created by export_model.py)
```

## Setup
```bash
pip install -r code/requirements.txt        # torch, torchvision, sklearn, ...
pip install -r app/requirements_api.txt      # fastapi, uvicorn, python-multipart, joblib
```

## 1. Export the model (once)
```bash
python app/export_model.py --palm_root "D:\palm_data\PALM"
# saves app/models/myopia_classifier.joblib and verifies held-out AUC
```

## 2. Run the server
```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

## API
| Method | Path           | Purpose                                  |
|--------|----------------|------------------------------------------|
| GET    | `/`            | Web frontend                             |
| GET    | `/api/health`  | Liveness + model-loaded flag             |
| GET    | `/api/info`    | Model card (backbone, protocol, metrics) |
| POST   | `/api/predict` | multipart image → myopia risk score      |

```bash
curl -F "file=@fundus.jpg" http://127.0.0.1:8000/api/predict
```

> The 1/2/3-year **onset** (clinical Cox) branch needs a longitudinal cohort, which
> is not public — the UI records biomarkers but never fabricates onset numbers.
> Research tool, not a medical device.

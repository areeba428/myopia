"""
=============================================================================
 INFERENCE  -  Score a fundus photograph for myopia risk (+ Grad-CAM)
 Backend: ONNX Runtime + NumPy + Pillow  (NO torch / sklearn / cv2 at runtime)
=============================================================================
Loads the ONNX ResNet-50 backbone (exported by export_onnx.py) and the linear
head stored in head_params.npz, then turns one image into a calibrated myopia
risk score in [0, 1]. Reproduces the genuine linear-eval scores (test AUC 0.988)
while keeping the runtime small enough for serverless deployment (~80 MB).

Grad-CAM
--------
The head is LINEAR: logit = w . standardize(GAP(layer4)) + b, so the Grad-CAM
weight for channel c is exactly alpha_c = w_c / scaler_scale_c, and the map is
ReLU( sum_c alpha_c * A_c ) over the layer4 maps A — exact, autograd-free.
=============================================================================
"""
from __future__ import annotations

import base64
import io
import os
import threading

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
ONNX_PATH = os.path.join(MODELS_DIR, "resnet50_features.onnx")
NPZ_PATH = os.path.join(MODELS_DIR, "head_params.npz")

_lock = threading.Lock()
_state = {"session": None, "p": None}

# Genuine held-out metrics from the linear-eval export (full PALM test set).
EVAL_METRICS = {"n_test": 400, "auc": 0.9879, "accuracy": 0.965,
                "sensitivity": 0.967, "specificity": 0.963}


def is_ready() -> bool:
    return _state["session"] is not None


def model_info() -> dict:
    if not is_ready():
        return {"loaded": False}
    return {
        "loaded": True,
        "backbone": "resnet50",
        "weights": "IMAGENET1K_V2",
        "protocol": "linear_eval",
        "runtime": "onnxruntime",
        "feature_dim": 2048,
        "eval_metrics": _eval_metrics(),
        "preprocess_mode": (_state["p"] or {}).get("preprocess_mode") or "legacy",
        "device": "cpu",
    }


def _eval_metrics() -> dict:
    """Held-out PALM metrics for the CURRENTLY served head (from the npz if present)."""
    p = _state["p"] or {}
    if "eval_auc" in p:
        return {"n_test": 400, "auc": round(float(p["eval_auc"]), 4),
                "accuracy": round(float(p["eval_acc"]), 4)}
    return EVAL_METRICS


def load_model():
    """Lazy, thread-safe, one-time load of the ONNX session + head params."""
    if is_ready():
        return
    with _lock:
        if is_ready():
            return
        if not (os.path.exists(ONNX_PATH) and os.path.exists(NPZ_PATH)):
            raise FileNotFoundError(
                f"Model artifacts missing in {MODELS_DIR}. "
                "Run:  python app/export_model.py --palm_root <PALM dir>  "
                "then  python app/export_onnx.py")
        import onnxruntime as ort

        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        npz = np.load(NPZ_PATH, allow_pickle=False)
        p = {k: npz[k] for k in npz.files}
        p["alpha"] = (p["coef"] / p["scaler_scale"]).astype(np.float32)  # Grad-CAM weights
        p["preprocess_mode"] = str(npz["preprocess_mode"]) if "preprocess_mode" in npz.files else None
        _state.update(session=sess, p=p)


# ---------------------------------------------------------------------------
# Preprocessing.
#   * preprocess_mode set  -> fundus-aware: FOV crop + colour normalization
#                             (domain-shift fix) -> 224 -> ImageNet normalize
#   * otherwise (legacy)   -> torchvision IMAGENET1K_V2: resize 232 -> crop 224
# ---------------------------------------------------------------------------
def _preprocess(img: Image.Image):
    """Return (model_input (1,C,H,W), vis_image) — vis is exactly what the net sees."""
    p = _state["p"]
    mode = p.get("preprocess_mode")
    if mode:
        from .fundus_preprocess import preprocess_image
        vis = preprocess_image(img, mode=mode, out_size=int(p["out_size"]))
    else:
        rs, cs = int(p["resize_size"]), int(p["crop_size"])
        w, h = img.size
        scale = rs / min(w, h)
        img_r = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        nw, nh = img_r.size
        left, top = (nw - cs) // 2, (nh - cs) // 2
        vis = img_r.crop((left, top, left + cs, top + cs))
    arr = np.asarray(vis, dtype=np.float32) / 255.0
    arr = (arr - p["norm_mean"]) / p["norm_std"]                 # HWC
    arr = np.transpose(arr, (2, 0, 1))[None, ...]               # 1,C,H,W
    return np.ascontiguousarray(arr, dtype=np.float32), vis


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _jet(v: np.ndarray) -> np.ndarray:
    """Vectorized JET colormap: v in [0,1] (H,W) -> uint8 RGB (H,W,3)."""
    f = 4.0 * v
    r = np.clip(np.minimum(f - 1.5, -f + 4.5), 0, 1)
    g = np.clip(np.minimum(f - 0.5, -f + 3.5), 0, 1)
    b = np.clip(np.minimum(f + 0.5, -f + 2.5), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _gradcam_overlay(orig: Image.Image, fmap: np.ndarray) -> str:
    """fmap: (2048, H, W) layer4 maps -> base64 PNG of heatmap blended on the image."""
    p = _state["p"]
    cam = np.tensordot(p["alpha"], fmap, axes=([0], [0]))       # (H, W)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    heat = Image.fromarray(_jet(cam)).resize(orig.size, Image.BILINEAR)
    base = np.asarray(orig, dtype=np.float32)
    blended = np.uint8(0.55 * base + 0.45 * np.asarray(heat, dtype=np.float32))
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _risk_band(prob: float) -> str:
    if prob >= 0.66:
        return "high"
    if prob >= 0.34:
        return "moderate"
    return "low"


def predict_bytes(data: bytes, heatmap: bool = False) -> dict:
    """Return the myopia risk score (+ optional Grad-CAM overlay) for one image."""
    load_model()
    p = _state["p"]
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x, vis = _preprocess(img)
    feat, fmap = _state["session"].run(["features", "fmap"], {"input": x})
    z = (feat[0] - p["scaler_mean"]) / p["scaler_scale"]        # standardize
    logit = float(np.dot(p["coef"], z) + p["intercept"])
    prob = float(_sigmoid(logit))

    label = int(prob >= 0.5)
    result = {
        "image_risk_score": round(prob, 4),
        "prediction": "pathologic myopia" if label else "non-pathologic",
        "label": label,
        "risk_band": _risk_band(prob),
        "confidence": round(float(abs(prob - 0.5) * 2), 4),
    }
    if heatmap:
        try:
            result["heatmap"] = _gradcam_overlay(vis, fmap[0])
        except Exception as e:
            result["heatmap_error"] = str(e)
    return result

"""
Select a preprocessing mode from the experiment and wire it into the served model.

Writes app/models/head_params.npz (used by inference.py) with the chosen mode's
linear head + the preprocessing flag, and refreshes myopia_classifier.joblib.
The ONNX backbone is unchanged, so no re-export is needed.

    python app/select_preprocess.py grayworld
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
EXPERIMENT_PATH = os.path.join(MODELS_DIR, "preprocess_experiment.joblib")
NPZ_PATH = os.path.join(MODELS_DIR, "head_params.npz")
JOBLIB_PATH = os.path.join(MODELS_DIR, "myopia_classifier.joblib")


def select(mode: str):
    exp = joblib.load(EXPERIMENT_PATH)
    if mode not in exp["heads"]:
        raise SystemExit(f"mode '{mode}' not in experiment. Have: {list(exp['heads'])}")
    scaler = exp["heads"][mode]["scaler"]
    clf = exp["heads"][mode]["classifier"]
    res = exp["results"][mode]

    np.savez(
        NPZ_PATH,
        coef=clf.coef_[0].astype(np.float32),
        intercept=np.float32(clf.intercept_[0]),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        norm_mean=exp["norm_mean"].astype(np.float32),
        norm_std=exp["norm_std"].astype(np.float32),
        out_size=np.int64(exp["out_size"]),
        preprocess_mode=np.str_(mode),
        eval_auc=np.float32(res["auc"]),
        eval_acc=np.float32(res["accuracy"]),
    )
    joblib.dump({"scaler": scaler, "classifier": clf, "backbone": "resnet50",
                 "weights": "IMAGENET1K_V2", "feature_dim": 2048,
                 "protocol": "linear_eval", "preprocess_mode": mode,
                 "eval_metrics": {"n_test": 400, **res}}, JOBLIB_PATH)
    print(f"Selected '{mode}' (PALM test AUC={res['auc']:.4f}, acc={res['accuracy']:.4f})")
    print(f"Wrote {NPZ_PATH} and {JOBLIB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python app/select_preprocess.py <mode>")
    select(sys.argv[1])

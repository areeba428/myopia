"""
=============================================================================
 EXPORT MODEL  -  Persist the trained myopia-detection head for inference
=============================================================================

The research `image_pipeline.py` re-fits the logistic-regression head every run
straight from PALM features (the "linear evaluation" protocol). For a real-world
API we need that head SAVED to disk so the server can score new fundus photos
without ever touching the PALM dataset again.

This script reproduces the genuine linear-evaluation training EXACTLY:
    1. ImageNet-pretrained ResNet-50 (fc -> Identity)  -> 2048 features / image
    2. Fit StandardScaler + LogisticRegression on PALM Training+Validation (800)
    3. (optional) verify on the held-out Testing split (400) -> should match the
       reported AUC = 0.988
    4. Save {scaler, classifier, preprocess metadata} to app/models/.

Run:
    python app/export_model.py --palm_root D:\palm_data\PALM
    python app/export_model.py --palm_root D:\palm_data\PALM --no_eval   (faster)
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Reuse the genuine, reviewed research code instead of re-implementing it.
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE_DIR)
from image_pipeline import (  # noqa: E402
    SEED, set_seed, load_split, extract_features,
)

MODELS_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "myopia_classifier.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palm_root", required=True)
    ap.add_argument("--no_eval", action="store_true",
                    help="Skip the held-out test evaluation (faster export).")
    args = ap.parse_args()
    set_seed()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train = load_split(args.palm_root, "Training")
    val = load_split(args.palm_root, "Validation")
    fit_df = pd.concat([train, val], ignore_index=True)
    print(f"Fitting head on Train+Val = {len(fit_df)} images "
          f"(positives={int(fit_df['label'].sum())})")

    print("Extracting ResNet-50 features for the fit set (CPU may take a few min)...")
    Xtr = extract_features(fit_df, device)
    ytr = fit_df["label"].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    print("Head fitted.")

    eval_metrics = None
    if not args.no_eval:
        from sklearn.metrics import accuracy_score, roc_auc_score
        test = load_split(args.palm_root, "Testing")
        print(f"Verifying on held-out Testing = {len(test)} images...")
        Xte = extract_features(test, device)
        yte = test["label"].to_numpy()
        prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
        eval_metrics = {
            "n_test": int(len(yte)),
            "auc": float(roc_auc_score(yte, prob)),
            "accuracy": float(accuracy_score(yte, (prob >= 0.5).astype(int))),
        }
        print(f"  Held-out AUC={eval_metrics['auc']:.3f} "
              f"accuracy={eval_metrics['accuracy']:.3f} "
              f"(report: AUC=0.988, acc=0.965)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        "scaler": scaler,
        "classifier": clf,
        "backbone": "resnet50",
        "weights": "IMAGENET1K_V2",
        "feature_dim": int(Xtr.shape[1]),
        "protocol": "linear_eval",
        "seed": SEED,
        "eval_metrics": eval_metrics,
    }, MODEL_PATH)
    print(f"\nSaved model artifact -> {MODEL_PATH}")
    if eval_metrics:
        with open(os.path.join(MODELS_DIR, "export_eval.json"), "w") as f:
            json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()

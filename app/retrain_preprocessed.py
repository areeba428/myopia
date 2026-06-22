"""
=============================================================================
 RETRAIN WITH FUNDUS PREPROCESSING  -  evidence-based domain-shift fix
=============================================================================
For each preprocessing mode, this:
  1. preprocesses every PALM image (FOV crop + colour normalization),
  2. extracts frozen ResNet-50 features,
  3. fits StandardScaler + LogisticRegression on Train+Val,
  4. evaluates on the held-out Testing split (the genuine metric).

It prints a per-mode table so we choose a mode that PRESERVES the PALM AUC
(~0.988) while normalizing away camera/colour shift. All fitted heads are saved
so the winner can be wired in without recomputing.

Run (offline, needs torch):
    python app/retrain_preprocessed.py --palm_root "D:\palm_data\PALM"
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
from image_pipeline import load_split, set_seed, SEED  # noqa: E402
from fundus_preprocess import MODES, preprocess_image  # noqa: E402

MODELS_DIR = os.path.join(HERE, "models")
EXPERIMENT_PATH = os.path.join(MODELS_DIR, "preprocess_experiment.joblib")
RESULTS_PATH = os.path.join(MODELS_DIR, "preprocess_experiment.json")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OUT_SIZE = 224


def build_backbone():
    import torch
    from torchvision import models
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    return net.eval()


def extract(df, mode, net, batch=16):
    """Preprocess (mode) -> ImageNet-normalize -> ResNet-50 features (N, 2048)."""
    import torch
    from PIL import Image
    feats, buf = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf))
        with torch.no_grad():
            feats.append(net(x).numpy())
        buf.clear()

    for i, p in enumerate(df["image_path"]):
        img = Image.open(p).convert("RGB")
        proc = preprocess_image(img, mode=mode, out_size=OUT_SIZE)
        a = (np.asarray(proc, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        buf.append(np.ascontiguousarray(np.transpose(a, (2, 0, 1)), dtype=np.float32))
        if len(buf) == batch:
            flush()
        if (i + 1) % 200 == 0:
            print(f"     {mode}: {i+1}/{len(df)}")
    flush()
    return np.concatenate(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palm_root", required=True)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    args = ap.parse_args()
    set_seed()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    train = load_split(args.palm_root, "Training")
    val = load_split(args.palm_root, "Validation")
    test = load_split(args.palm_root, "Testing")
    import pandas as pd
    fit_df = pd.concat([train, val], ignore_index=True)
    ytr, yte = fit_df["label"].to_numpy(), test["label"].to_numpy()

    net = build_backbone()
    os.makedirs(MODELS_DIR, exist_ok=True)
    saved, results = {}, {}
    print(f"{'mode':<12}{'test AUC':>10}{'accuracy':>10}")
    print("-" * 32)
    for mode in args.modes:
        print(f"[{mode}] extracting Train+Val ({len(fit_df)})...")
        Xtr = extract(fit_df, mode, net)
        print(f"[{mode}] extracting Test ({len(test)})...")
        Xte = extract(test, mode, net)
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(scaler.transform(Xtr), ytr)
        prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
        auc = float(roc_auc_score(yte, prob))
        acc = float(accuracy_score(yte, (prob >= 0.5).astype(int)))
        saved[mode] = {"scaler": scaler, "classifier": clf}
        results[mode] = {"auc": auc, "accuracy": acc}
        print(f"{mode:<12}{auc:>10.4f}{acc:>10.4f}")

    joblib.dump({"heads": saved, "out_size": OUT_SIZE,
                 "norm_mean": IMAGENET_MEAN, "norm_std": IMAGENET_STD,
                 "results": results}, EXPERIMENT_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    best = max(results, key=lambda m: results[m]["auc"])
    print("-" * 32)
    print(f"Best PALM AUC: '{best}' ({results[best]['auc']:.4f})")
    print(f"Saved all heads -> {EXPERIMENT_PATH}")
    print("Next: select a mode -> python app/select_preprocess.py <mode>")


if __name__ == "__main__":
    main()

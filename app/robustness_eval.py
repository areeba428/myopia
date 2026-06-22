"""
=============================================================================
 ROBUSTNESS EVALUATION  -  evidence that preprocessing fixes domain shift
=============================================================================
Applies synthetic camera/domain shifts (colour casts, brightness, contrast) to
the PALM Testing split and compares the LEGACY model (resize+crop, ImageNet) vs
the new CLAHE-preprocessed model, on two metrics:

  * AUC                : discrimination under the shift (higher = better)
  * mean score on TRUE-NORMAL images : the false-positive tendency
                         (your exact symptom — normal fundi scored as myopia).
                         Lower-and-stable under shift = the fix is working.

Run (offline, needs torch):
    python app/robustness_eval.py --palm_root "D:\palm_data\PALM"
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
from image_pipeline import load_split, set_seed  # noqa: E402
from fundus_preprocess import preprocess_image  # noqa: E402

MODELS_DIR = os.path.join(HERE, "models")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def corrupt(img: Image.Image, kind: str) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    if kind == "identity":
        pass
    elif kind == "warm_cast":
        a[..., 0] *= 1.20; a[..., 2] *= 0.80
    elif kind == "cool_cast":
        a[..., 0] *= 0.80; a[..., 2] *= 1.20
    elif kind == "bright":
        a *= 1.35
    elif kind == "dark":
        a *= 0.65
    elif kind == "low_contrast":
        a = 128 + (a - 128) * 0.5
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def tv_transform(img: Image.Image) -> np.ndarray:
    """Legacy torchvision IMAGENET1K_V2: resize 232 -> center-crop 224 -> normalize."""
    w, h = img.size
    s = 232 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    nw, nh = img.size
    l, t = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    a = (np.asarray(img, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1)), dtype=np.float32)


def clahe_transform(img: Image.Image) -> np.ndarray:
    proc = preprocess_image(img, mode="clahe", out_size=224)
    a = (np.asarray(proc, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1)), dtype=np.float32)


def features(net, tensors, batch=16):
    import torch
    out = []
    for i in range(0, len(tensors), batch):
        x = torch.from_numpy(np.stack(tensors[i:i + batch]))
        with torch.no_grad():
            out.append(net(x).numpy())
    return np.concatenate(out, 0)


def head_prob(bundle, feats):
    z = (feats - bundle["scaler"].mean_) / bundle["scaler"].scale_
    return bundle["classifier"].predict_proba(z)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palm_root", required=True)
    args = ap.parse_args()
    set_seed()
    from sklearn.metrics import roc_auc_score
    from torchvision import models
    import torch

    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity(); net.eval()

    legacy = joblib.load(os.path.join(MODELS_DIR, "myopia_classifier_legacy.joblib"))
    clahe = joblib.load(os.path.join(MODELS_DIR, "myopia_classifier.joblib"))

    test = load_split(args.palm_root, "Testing")
    y = test["label"].to_numpy()
    neg = y == 0
    imgs = [Image.open(p).convert("RGB") for p in test["image_path"]]

    kinds = ["identity", "warm_cast", "cool_cast", "bright", "dark", "low_contrast"]
    rows, results = [], {}
    hdr = f"{'corruption':<14}{'LEGACY auc':>12}{'CLAHE auc':>11}{'LEG neg-score':>15}{'CLAHE neg-score':>17}"
    print(hdr); print("-" * len(hdr))
    for k in kinds:
        cor = [corrupt(im, k) for im in imgs]
        lp = head_prob(legacy, features(net, [tv_transform(c) for c in cor]))
        cp = head_prob(clahe, features(net, [clahe_transform(c) for c in cor]))
        la, ca = roc_auc_score(y, lp), roc_auc_score(y, cp)
        ln, cn = float(lp[neg].mean()), float(cp[neg].mean())
        results[k] = {"legacy_auc": float(la), "clahe_auc": float(ca),
                      "legacy_neg_meanscore": ln, "clahe_neg_meanscore": cn}
        print(f"{k:<14}{la:>12.4f}{ca:>11.4f}{ln:>15.3f}{cn:>17.3f}")

    with open(os.path.join(MODELS_DIR, "robustness_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> app/models/robustness_results.json")
    print("Lower & flatter 'neg-score' across corruptions = fewer false positives on normal fundi.")


if __name__ == "__main__":
    main()

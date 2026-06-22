"""
Per-image BEFORE/AFTER on your own fundus photos.

Drop images into  My_Study_Final/samples/  then run:
    python app/compare_samples.py

For each image it prints the legacy vs CLAHE myopia score and writes a
side-by-side panel (original | legacy Grad-CAM | CLAHE Grad-CAM) to
samples/_compare/. Uses the ONNX backbone — no torch needed.
"""
from __future__ import annotations

import glob
import os

import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
SAMPLES = os.path.join(HERE, "..", "samples")
ONNX = os.path.join(MODELS, "resnet50_features.onnx")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

import sys
sys.path.insert(0, HERE)
from fundus_preprocess import preprocess_image  # noqa: E402


def tv_vis(img):
    w, h = img.size
    s = 232 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    nw, nh = img.size
    l, t = (nw - 224) // 2, (nh - 224) // 2
    return img.crop((l, t, l + 224, t + 224))


def norm(vis):
    a = (np.asarray(vis, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1))[None], np.float32)


def jet(v):
    f = 4 * v
    r = np.clip(np.minimum(f - 1.5, -f + 4.5), 0, 1)
    g = np.clip(np.minimum(f - 0.5, -f + 3.5), 0, 1)
    b = np.clip(np.minimum(f + 0.5, -f + 2.5), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def run(sess, head, vis):
    feat, fmap = sess.run(["features", "fmap"], {"input": norm(vis)})
    z = (feat[0] - head["scaler"].mean_) / head["scaler"].scale_
    prob = float(head["classifier"].predict_proba(z[None])[0, 1])
    alpha = head["classifier"].coef_[0] / head["scaler"].scale_
    cam = np.maximum(np.tensordot(alpha, fmap[0], axes=([0], [0])), 0)
    if cam.max() > 0:
        cam /= cam.max()
    heat = Image.fromarray(jet(cam)).resize(vis.size, Image.BILINEAR)
    blend = np.uint8(0.55 * np.asarray(vis, np.float32) + 0.45 * np.asarray(heat, np.float32))
    return prob, Image.fromarray(blend)


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 255))
    return img


def main():
    files = sorted(sum([glob.glob(os.path.join(SAMPLES, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")], []))
    if not files:
        print(f"No images in {os.path.abspath(SAMPLES)} — drop your fundus photos there first.")
        return
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    legacy = joblib.load(os.path.join(MODELS, "myopia_classifier_legacy.joblib"))
    clahe = joblib.load(os.path.join(MODELS, "myopia_classifier.joblib"))
    out_dir = os.path.join(SAMPLES, "_compare")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'image':<28}{'LEGACY':>9}{'CLAHE':>9}")
    print("-" * 46)
    for f in files:
        img = Image.open(f).convert("RGB")
        lp, lh = run(sess, legacy, tv_vis(img))
        cp, ch = run(sess, clahe, preprocess_image(img, mode="clahe", out_size=224))
        print(f"{os.path.basename(f):<28}{lp:>9.3f}{cp:>9.3f}")
        panel = Image.new("RGB", (224 * 3 + 8, 224), (20, 20, 20))
        panel.paste(label(tv_vis(img).copy(), "original"), (0, 0))
        panel.paste(label(lh, f"legacy {lp:.2f}"), (228, 0))
        panel.paste(label(ch, f"clahe {cp:.2f}"), (456, 0))
        panel.save(os.path.join(out_dir, "cmp_" + os.path.splitext(os.path.basename(f))[0] + ".png"))
    print(f"\nSide-by-side panels saved to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()

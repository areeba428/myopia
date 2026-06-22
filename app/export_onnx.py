"""
=============================================================================
 EXPORT ONNX  -  Make the image branch deployable WITHOUT PyTorch
=============================================================================
PyTorch (~1.3 GB) cannot fit in a Vercel serverless function (250 MB limit).
This script converts the frozen ResNet-50 backbone to ONNX and dumps the
linear head (logistic regression + scaler) to a tiny .npz, so the server can
run on  onnxruntime + numpy + Pillow  (~80 MB total) and reproduce the exact
same scores.

Outputs (in app/models/):
    resnet50_features.onnx   two outputs:
                               - "features" : (N, 2048) global-avg-pooled vector
                               - "fmap"     : (N, 2048, 7, 7) layer4 maps (Grad-CAM)
    head_params.npz          coef, intercept, scaler mean/scale, preprocess cfg

Run (needs torch, one time, on any machine):
    python app/export_onnx.py
=============================================================================
"""
from __future__ import annotations

import os

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
JOBLIB_PATH = os.path.join(MODELS_DIR, "myopia_classifier.joblib")
ONNX_PATH = os.path.join(MODELS_DIR, "resnet50_features.onnx")
NPZ_PATH = os.path.join(MODELS_DIR, "head_params.npz")


def main():
    import torch
    import torch.nn as nn
    from torchvision import models

    if not os.path.exists(JOBLIB_PATH):
        raise SystemExit(f"Run export_model.py first to create {JOBLIB_PATH}")
    bundle = joblib.load(JOBLIB_PATH)
    scaler, clf = bundle["scaler"], bundle["classifier"]

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    base = models.resnet50(weights=weights).eval()

    class Backbone(nn.Module):
        """Return both the pooled feature vector and the layer4 feature maps."""
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            n = self.net
            x = n.conv1(x); x = n.bn1(x); x = n.relu(x); x = n.maxpool(x)
            x = n.layer1(x); x = n.layer2(x); x = n.layer3(x)
            fmap = n.layer4(x)                 # (N, 2048, 7, 7)
            feat = n.avgpool(fmap).flatten(1)  # (N, 2048)
            return feat, fmap

    model = Backbone(base).eval()
    dummy = torch.randn(1, 3, 224, 224)

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["features", "fmap"],
        dynamic_axes={"input": {0: "n"}, "features": {0: "n"}, "fmap": {0: "n"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"Saved ONNX backbone -> {ONNX_PATH} ({os.path.getsize(ONNX_PATH)/1e6:.0f} MB)")

    # The torchvision IMAGENET1K_V2 transform: resize 232 (bilinear) -> center-crop 224
    # -> scale [0,1] -> normalize. We persist the exact constants for the numpy path.
    np.savez(
        NPZ_PATH,
        coef=clf.coef_[0].astype(np.float32),
        intercept=np.float32(clf.intercept_[0]),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        resize_size=np.int64(232),
        crop_size=np.int64(224),
        norm_mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
        norm_std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
    )
    print(f"Saved head params -> {NPZ_PATH}")


if __name__ == "__main__":
    main()

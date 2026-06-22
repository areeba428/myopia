"""
=============================================================================
 FUNDUS PREPROCESSING  -  make the image branch robust to camera/domain shift
=============================================================================
The model was trained on PALM (one camera/population). Real-world photos differ
in field of view, illumination and colour balance, which shifts the ImageNet
features and causes false positives. This module normalizes those nuisance
factors BEFORE the backbone, applied identically at train and inference time.

Pure NumPy + Pillow (no OpenCV/torch) so it runs both offline (retraining) and
in the slim serverless inference path.

Pipeline
    1. FOV detection + tight crop   -> removes the black border, normalizes the
                                       visible-retina field of view across cameras
    2. pad to square + resize       -> consistent geometry, no peripheral loss
    3. colour / illumination norm   -> one of:
         "none"      : geometry only (FOV crop + resize)
         "grayworld" : gray-world colour constancy (per-channel mean equalised)
         "clahe"     : grayworld + global histogram contrast stretch
         "graham"    : Ben Graham local-mean subtraction (Kaggle DR 2015 winner)
    4. circular mask                -> zero the corners outside the FOV
=============================================================================
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

MODES = ("none", "grayworld", "clahe", "graham")


def _crop_to_fov(arr: np.ndarray):
    """Crop to the bounding box of the illuminated (non-black) fundus region."""
    gray = arr.mean(axis=2)
    thr = max(7.0, float(gray.max()) * 0.06)
    mask = gray > thr
    if not mask.any():
        return arr
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return arr[y0:y1 + 1, x0:x1 + 1]


def _pad_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=arr.dtype)
    oy, ox = (s - h) // 2, (s - w) // 2
    out[oy:oy + h, ox:ox + w] = arr
    return out


def _circle_mask(size: int) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2.0
    r = size / 2.0
    return ((yy - c) ** 2 + (xx - c) ** 2) <= (r * r)


def _grayworld(arr: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Scale each channel so its in-FOV mean matches the overall mean (colour constancy)."""
    px = arr[fov]
    means = px.mean(axis=0) + 1e-6
    target = float(means.mean())
    return arr * (target / means)


def preprocess_array(arr: np.ndarray, mode: str = "grayworld", out_size: int = 224) -> np.ndarray:
    """RGB uint8 array -> preprocessed RGB uint8 array (out_size x out_size)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    arr = _crop_to_fov(arr.astype(np.uint8))
    arr = _pad_square(arr)
    img = Image.fromarray(arr).resize((out_size, out_size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)

    fov = _circle_mask(out_size)
    if mode in ("grayworld", "clahe"):
        a = _grayworld(a, fov)
    if mode == "clahe":
        # global contrast stretch on in-FOV pixels (a light, CLAHE-like normalization)
        px = a[fov]
        lo, hi = np.percentile(px, 1), np.percentile(px, 99)
        if hi > lo:
            a = (a - lo) / (hi - lo) * 255.0
    if mode == "graham":
        blur = img.filter(ImageFilter.GaussianBlur(radius=out_size / 30.0))
        a = 4.0 * (a - np.asarray(blur, dtype=np.float32)) + 128.0

    a = np.clip(a, 0, 255)
    a[~fov] = 0  # black out the corners outside the circular FOV
    return a.astype(np.uint8)


def preprocess_image(img: Image.Image, mode: str = "grayworld", out_size: int = 224) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    return Image.fromarray(preprocess_array(arr, mode=mode, out_size=out_size))

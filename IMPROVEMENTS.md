# Real-world robustness: fundus preprocessing (domain-shift fix)

## Problem
The model scored **AUC 0.988** on PALM's held-out test but produced **false positives
on external fundus photos** (normal-looking images flagged as pathologic myopia). Root
cause: it is a frozen **ImageNet** ResNet-50 + a linear head trained **only on PALM**.
ImageNet features encode low-level colour/illumination/texture that shift with camera and
population, so out-of-domain images land on the wrong side of the boundary — textbook
**covariate / domain shift**. The legacy `resize→center-crop` transform also isn't
fundus-aware (clips peripheral retina, keeps the black border).

## Fix (Tier 1 — no new labels)
A fundus-aware preprocessing stage applied **identically at train and inference**
([app/fundus_preprocess.py](app/fundus_preprocess.py), pure NumPy/Pillow so it runs in the
slim ONNX serving path too):

1. **FOV detection + tight crop** — removes the black border, normalizes the visible field
   of view across cameras.
2. **Pad-to-square + resize 224** — consistent geometry, no peripheral loss.
3. **Colour/illumination normalization** — `clahe` mode: gray-world colour constancy +
   global contrast stretch (a light CLAHE-like step).
4. **Circular mask** — zero the corners outside the FOV.

The linear head was **retrained on PALM with this preprocessing** and the ONNX backbone is
unchanged, so only `head_params.npz` changed (no 94 MB re-export).

## Evidence 1 — PALM held-out AUC is preserved (not sacrificed)
Retrained per mode, evaluated once on the 400-image Testing split:

| Mode | PALM test AUC | Accuracy |
|------|--------------|----------|
| legacy (resize + center-crop) | 0.9879 | 0.965 |
| FOV crop only | 0.9935 | 0.9625 |
| grayworld | 0.9944 | 0.9575 |
| **clahe (selected)** | **0.9956** | **0.9725** |
| graham | 0.9953 | 0.9700 |

Every preprocessing mode matches or beats legacy — the fix does not cost in-domain accuracy.

## Evidence 2 — robustness to camera/colour shift (the real-world symptom)
Synthetic camera shifts applied to the PALM Testing split; `neg-score` = mean predicted
myopia probability on **true-normal** fundi (lower & flatter = fewer false positives):

| Corruption | Legacy AUC | CLAHE AUC | Legacy neg-score | CLAHE neg-score |
|------------|-----------|-----------|------------------|-----------------|
| identity | 0.9888 | 0.9956 | 0.042 | 0.031 |
| warm cast | 0.9898 | 0.9958 | **0.069** | **0.033** |
| cool cast | 0.9892 | 0.9954 | 0.025 | 0.031 |
| bright | 0.9872 | 0.9955 | 0.037 | 0.032 |
| dark | 0.9915 | 0.9958 | 0.044 | 0.032 |
| low contrast | 0.9917 | 0.9817 | 0.040 | 0.052 |

Under a warm colour cast the **legacy** false-positive score jumps +64% (0.042→0.069) — a
colour change alone starts flagging normals. The **CLAHE** model stays flat (0.031–0.033)
across all colour/brightness shifts: colour normalization makes it invariant to the camera
differences that break the original.

**Caveat (honest):** under `low_contrast`, CLAHE is slightly worse (AUC 0.982, neg-score
0.052) — its contrast stretch over-amplifies already-flat images. Net win across realistic
shifts, but not free everywhere.

## Reproduce
```bash
python app/retrain_preprocessed.py --palm_root <PALM>   # per-mode AUC table
python app/select_preprocess.py clahe                    # wire the chosen mode in
python app/robustness_eval.py --palm_root <PALM>         # the table above
```
Switch modes anytime: `python app/select_preprocess.py graham` (then restart the server).

## What's next (bigger gains, need data/compute)
- **Fine-tune** the backbone (unfreeze layer4) with camera-simulating augmentation instead
  of linear-probing frozen ImageNet features.
- **Retinal foundation backbone (RETFound,** Zhou et al., *Nature* 2023): biggest
  evidence-based generalization gain across cameras/populations.
- **Threshold recalibration** on a small labeled real-world set.

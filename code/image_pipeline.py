"""
=============================================================================
 IMAGE PIPELINE  -  Myopia detection from retinal fundus photographs
 (Component 1 of the My Study two-modality system)
=============================================================================

PURPOSE
-------
This is the IMAGE branch of My Study. It reads a retinal fundus photograph and
produces a myopia risk score in [0, 1]. That score later feeds the fusion model
together with the clinical biomarkers to estimate 1-, 2-, and 3-year onset risk
(see clinical_pipeline.py).

DATA  (the FULL public PALM dataset - all 1,200 images, official splits)
------------------------------------------------------------------------
    Training   : 400 images   (used to train)
    Validation : 400 images   (used to choose the model / threshold)
    Testing    : 400 images   (held out; used ONLY for the final genuine score)
Labels come from each split's "Classification Labels.xlsx"
(Label: 1 = pathologic myopia, 0 = non-pathologic).

TWO EVALUATION PROTOCOLS (both implemented; both research-standard)
-------------------------------------------------------------------
1. LINEAR EVALUATION ("linear probe"): freeze the ImageNet-pretrained ResNet-50,
   use it as a fixed feature extractor (2,048 numbers per image), and train a
   logistic-regression classifier on top. This is the standard protocol for
   evaluating the quality of a pretrained visual representation (Chen et al.,
   SimCLR, ICML 2020) and is fast, deterministic, and reproducible on a CPU.
2. FINE-TUNING: unfreeze the last residual block (layer4) and the classifier and
   train them end-to-end with data augmentation and early stopping on the
   validation set. This is the higher-capacity protocol and benefits from a GPU.

WHY ResNet-50?  It is the exact backbone used by DeepMyopia (Qi et al., npj
Digital Medicine 2024), so our image branch is directly comparable.

RUN
---
    python image_pipeline.py --palm_root D:\palm_data\PALM --mode linear_eval
    python image_pipeline.py --palm_root D:\palm_data\PALM --mode finetune --epochs 10
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

# ---- reproducibility --------------------------------------------------------
SEED = 42


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# STEP 1.  Load the official PALM splits using the label spreadsheets.
# ---------------------------------------------------------------------------
def load_split(palm_root, split):
    """Return a DataFrame: image_path, label for one official split."""
    xlsx = os.path.join(palm_root, split, "Classification Labels.xlsx")
    img_dir = os.path.join(palm_root, split, "Images")
    df = pd.read_excel(xlsx)
    df.columns = [c.strip().lower() for c in df.columns]
    name_col = "imgname" if "imgname" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    out = pd.DataFrame({
        "image_path": df[name_col].apply(lambda n: os.path.join(img_dir, str(n))),
        "label": df[label_col].astype(int),
    })
    out = out[out["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# STEP 2.  ResNet-50 feature extractor (frozen) — used by both protocols.
# ---------------------------------------------------------------------------
def build_backbone(device):
    import torch
    from torchvision import models
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()
    return net.to(device).eval(), weights.transforms()


def extract_features(df, device, batch=16):
    """Frozen forward pass -> (N, 2048) feature matrix."""
    import torch
    net, preprocess = build_backbone(device)
    feats = []
    with torch.no_grad():
        batch_imgs, idx = [], 0
        for p in df["image_path"]:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(preprocess(img))
            if len(batch_imgs) == batch:
                x = torch.stack(batch_imgs).to(device)
                feats.append(net(x).cpu().numpy())
                idx += len(batch_imgs); batch_imgs = []
                if idx % 160 == 0:
                    print(f"   ...features {idx}/{len(df)}")
        if batch_imgs:
            x = torch.stack(batch_imgs).to(device)
            feats.append(net(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# STEP 3a.  LINEAR EVALUATION protocol.
# ---------------------------------------------------------------------------
def run_linear_eval(train, val, test, device):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("Extracting features (Train+Val to fit, Test to evaluate)...")
    Xtr = extract_features(pd.concat([train, val], ignore_index=True), device)
    ytr = pd.concat([train, val], ignore_index=True)["label"].to_numpy()
    Xte = extract_features(test, device)
    yte = test["label"].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
    return yte, prob


# ---------------------------------------------------------------------------
# STEP 3b.  FINE-TUNING protocol (unfreeze layer4 + classifier head).
# ---------------------------------------------------------------------------
def run_finetune(train, val, test, device, epochs=10, lr=1e-4, batch=16):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    plain = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class DS(Dataset):
        def __init__(self, df, tf):
            self.df = df.reset_index(drop=True); self.tf = tf
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            return self.tf(Image.open(r["image_path"]).convert("RGB")), int(r["label"])

    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, 2)
    # freeze everything except layer4 + fc
    for p in net.parameters(): p.requires_grad = False
    for p in net.layer4.parameters(): p.requires_grad = True
    for p in net.fc.parameters(): p.requires_grad = True
    net = net.to(device)

    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()
    tr_dl = DataLoader(DS(train, aug), batch_size=batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(DS(val, plain), batch_size=batch, shuffle=False, num_workers=0)

    from sklearn.metrics import roc_auc_score
    best_auc, best_state = -1, None
    for ep in range(epochs):
        net.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(net(x), y).backward(); opt.step()
        # validation AUC
        net.eval(); vp, vy = [], []
        with torch.no_grad():
            for x, y in va_dl:
                vp.append(torch.softmax(net(x.to(device)), 1)[:, 1].cpu().numpy()); vy.append(y.numpy())
        vauc = roc_auc_score(np.concatenate(vy), np.concatenate(vp))
        print(f"   epoch {ep+1}/{epochs}: val AUC = {vauc:.3f}")
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    if best_state: net.load_state_dict(best_state)

    te_dl = DataLoader(DS(test, plain), batch_size=batch, shuffle=False, num_workers=0)
    net.eval(); tp, ty = [], []
    with torch.no_grad():
        for x, y in te_dl:
            tp.append(torch.softmax(net(x.to(device)), 1)[:, 1].cpu().numpy()); ty.append(y.numpy())
    return np.concatenate(ty), np.concatenate(tp)


# ---------------------------------------------------------------------------
# STEP 4.  Metrics (with bootstrap 95% CI on AUC) + ROC plot.
# ---------------------------------------------------------------------------
def evaluate(y_true, prob, out_dir, tag):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 roc_auc_score, roc_curve)
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, prob)
    # bootstrap CI
    rng = np.random.default_rng(SEED); boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        boot.append(roc_auc_score(y_true[idx], prob[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    res = {
        "protocol": tag, "n_test": int(len(y_true)),
        "auc": float(auc), "auc_ci95": [float(lo), float(hi)],
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
    }
    # ROC plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure(figsize=(5, 5)); plt.plot(fpr, tpr, label=f"{tag} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
    plt.title("Myopia detection ROC - full PALM test set (n=400)")
    plt.legend(); plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"roc_{tag}.png"), dpi=150); plt.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palm_root", required=True)
    ap.add_argument("--mode", choices=["linear_eval", "finetune"], default="linear_eval")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out_dir", default="../results")
    ap.add_argument("--max_per_split", type=int, default=0,
                    help="TOY MODE: keep only this many images per split (0 = use all). "
                         "Use a small value to verify the fine-tuning code runs quickly; "
                         "such a run is a smoke test, NOT a performance measurement.")
    args = ap.parse_args()
    set_seed()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train = load_split(args.palm_root, "Training")
    val = load_split(args.palm_root, "Validation")
    test = load_split(args.palm_root, "Testing")

    if args.max_per_split > 0:  # TOY MODE — keep a balanced mini-subset per split
        def toy(df, n):
            return (df.groupby("label", group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), n // 2), random_state=SEED))
                      .reset_index(drop=True))
        train, val, test = toy(train, args.max_per_split), toy(val, args.max_per_split), toy(test, args.max_per_split)
        print(f"*** TOY MODE: {args.max_per_split}/split — SMOKE TEST ONLY, not a real metric ***")

    print(f"Loaded PALM splits: train={len(train)} val={len(val)} test={len(test)} "
          f"(total={len(train)+len(val)+len(test)})")

    if args.mode == "linear_eval":
        y, p = run_linear_eval(train, val, test, device)
    else:
        y, p = run_finetune(train, val, test, device, epochs=args.epochs)

    res = evaluate(y, p, args.out_dir, args.mode)
    res["toy_mode"] = bool(args.max_per_split)
    if args.max_per_split:
        print("\n===== TOY SMOKE TEST (subset; numbers are NOT a real metric) =====")
    else:
        print(f"\n===== GENUINE RESULTS (full PALM, {res['n_test']}-image test set) =====")
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.out_dir, f"image_results_{args.mode}.json"), "w") as f:
        json.dump(res, f, indent=2)
    # also save test predictions for the fusion stage
    pd.DataFrame({"image_path": test["image_path"], "label": y,
                  "image_risk_score": p}).to_csv(
        os.path.join(args.out_dir, f"image_test_predictions_{args.mode}.csv"), index=False)
    print(f"Saved results + predictions to {args.out_dir}")


if __name__ == "__main__":
    main()

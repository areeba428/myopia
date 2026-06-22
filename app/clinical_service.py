"""
=============================================================================
 CLINICAL SERVICE  -  1/2/3-year onset risk via a fitted Cox model
=============================================================================
Wraps the genuine clinical_pipeline.py logic for the API. It NEVER fabricates
onset numbers: until a real longitudinal cohort CSV is fitted, prediction is
disabled and the service reports "data collection ongoing".

Flow
    fit_from_csv(path)  -> fits CoxPHFitter on the cohort, PERSISTS it
    is_ready()          -> whether a cohort model has been fitted
    predict_one(dict)   -> P(onset within 1, 2, 3 yr) for one new child
=============================================================================
"""
from __future__ import annotations

import os
import sys
import threading

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from clinical_pipeline import (  # noqa: E402
    BIOMARKERS, HORIZONS, REQUIRED, fit_onset_model, treatment_response,
)

MODELS_DIR = os.path.join(HERE, "models")
COX_PATH = os.path.join(MODELS_DIR, "cox_onset_model.joblib")

_lock = threading.Lock()
_state = {"cph": None, "features": None, "meta": None}

ALL_INPUTS = BIOMARKERS + ["image_risk_score"]


def is_ready() -> bool:
    return _state["cph"] is not None


def _try_load_persisted():
    if _state["cph"] is None and os.path.exists(COX_PATH):
        b = joblib.load(COX_PATH)
        _state.update(cph=b["cph"], features=b["features"], meta=b["meta"])


def status() -> dict:
    _try_load_persisted()
    if not is_ready():
        return {
            "ready": False,
            "message": "Clinical onset model not fitted — data collection ongoing. "
                       "Fit a longitudinal cohort CSV at POST /api/clinical/fit.",
            "required_columns": REQUIRED + BIOMARKERS + ["image_risk_score", "treatment"],
        }
    return {"ready": True, **_state["meta"]}


def fit_from_csv(path: str) -> dict:
    """Fit + persist the Cox onset model from a cohort CSV. Returns a fit report."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    feature_cols = [c for c in ALL_INPUTS if c in df.columns]
    if not feature_cols:
        raise ValueError(f"CSV has none of the predictor columns: {ALL_INPUTS}")

    with _lock:
        cph = fit_onset_model(df, feature_cols)
        meta = {
            "n": int(len(df)),
            "features": feature_cols,
            "c_index": float(cph.concordance_index_),
            "hazard_ratios": {k: float(v) for k, v in cph.hazard_ratios_.to_dict().items()},
            "horizons_years": HORIZONS,
        }
        if "treatment" in df.columns and df["treatment"].nunique() > 1:
            try:
                meta["treatment_response"] = treatment_response(df)
            except Exception as e:
                meta["treatment_response_error"] = str(e)

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({"cph": cph, "features": feature_cols, "meta": meta}, COX_PATH)
        _state.update(cph=cph, features=feature_cols, meta=meta)
    return meta


def predict_one(values: dict) -> dict:
    """P(onset within 1, 2, 3 years) for one child given biomarkers (+ image score)."""
    _try_load_persisted()
    if not is_ready():
        raise RuntimeError(
            "Clinical onset model not fitted (data collection ongoing). "
            "POST a cohort CSV to /api/clinical/fit first.")
    feats = _state["features"]
    row = {c: values.get(c, np.nan) for c in feats}
    X = pd.DataFrame([row]).fillna(0)  # real cohorts should supply all biomarkers
    surv = _state["cph"].predict_survival_function(X, times=HORIZONS)
    probs = (1 - surv.T).iloc[0]
    out = {f"onset_{t}yr_prob": round(float(probs.iloc[i]), 4)
           for i, t in enumerate(HORIZONS)}
    out["used_features"] = feats
    return out

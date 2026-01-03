"""Inductive conformal prediction for multi-class classification.

Nonconformity score: s(x, y) = 1 - P_hat(y | x).
For a desired miscoverage alpha, set qhat = Quantile_{ceil((n+1)*(1-alpha))/n} of scores on calibration set.
Prediction set: { y : 1 - P_hat(y|x) <= qhat }  <=>  P_hat(y|x) >= 1 - qhat.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import Config
from .utils import load_dataset, ensure_dir, save_json


def quantile_plus(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile with the +1 correction (finite-sample)."""
    n = scores.shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/Crop_recommendation.csv")
    ap.add_argument("--alpha", type=float, default=0.1, help="Target miscoverage (e.g., 0.1 for ~90% coverage)")
    ap.add_argument("--out", default="results/metrics/conformal.json")
    args = ap.parse_args()

    cfg = Config()
    X, y = load_dataset(args.data, cfg)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    # Split off final test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_enc, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y_enc
    )
    # Split training into proper-train and calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=cfg.calib_size, random_state=cfg.random_state, stratify=y_train_full
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            max_features=cfg.max_features,
        ))
    ])
    model.fit(X_train, y_train)

    # Calibration scores
    p_cal = model.predict_proba(X_cal)
    s_cal = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    qhat = quantile_plus(s_cal, args.alpha)
    threshold = 1.0 - qhat

    # Test prediction sets
    p_test = model.predict_proba(X_test)
    pred_sets = p_test >= threshold

    set_sizes = pred_sets.sum(axis=1)
    covered = pred_sets[np.arange(len(y_test)), y_test]
    coverage = float(covered.mean())
    avg_size = float(set_sizes.mean())
    med_size = float(np.median(set_sizes))
    empty_rate = float((set_sizes == 0).mean())

    report = {
        "alpha": args.alpha,
        "qhat": qhat,
        "prob_threshold": threshold,
        "coverage": coverage,
        "avg_set_size": avg_size,
        "median_set_size": med_size,
        "empty_set_rate": empty_rate,
        "n_test": int(len(y_test)),
        "n_classes": int(n_classes),
        "note": "Empty sets can occur when all class probabilities are below the threshold. Treat as abstentions or add a fallback top-1 rule in deployment."
    }

    ensure_dir(Path(args.out).parent)
    save_json(report, args.out)
    print(json_pretty(report))
    print("\nSaved:", args.out)


def json_pretty(d):
    import json
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    main()

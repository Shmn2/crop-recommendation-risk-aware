"""Calibrate probabilities (sigmoid/isotonic) and report ECE before/after."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .config import Config
from .utils import load_dataset, ensure_dir, ece_score, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/Crop_recommendation.csv")
    ap.add_argument("--method", choices=["sigmoid", "isotonic"], default="isotonic")
    ap.add_argument("--out", default="results/metrics/calibration.json")
    args = ap.parse_args()

    cfg = Config()
    X, y = load_dataset(args.data, cfg)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_enc
    )

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ExtraTreesClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            max_features=cfg.max_features,
        ))
    ])
    base.fit(X_train, y_train)
    probs_before = base.predict_proba(X_test)

    # Calibrate using an internal CV over the training data (practical + simple)
    calibrated = CalibratedClassifierCV(base, method=args.method, cv=3)
    calibrated.fit(X_train, y_train)
    probs_after = calibrated.predict_proba(X_test)

    report = {
        "method": args.method,
        "ece_before": ece_score(probs_before, y_test),
        "ece_after": ece_score(probs_after, y_test),
        "logloss_before": float(log_loss(y_test, probs_before)),
        "logloss_after": float(log_loss(y_test, probs_after)),
    }

    ensure_dir(Path(args.out).parent)
    save_json(report, args.out)
    print(json_pretty(report))
    print("\nSaved:", args.out)

    # Save calibrated model artifacts (optional, useful for deployment)
    ensure_dir("models")
    joblib.dump(calibrated, "models/extra_trees_calibrated.joblib")
    joblib.dump(le, "models/label_encoder.joblib")


def json_pretty(d):
    import json
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    main()

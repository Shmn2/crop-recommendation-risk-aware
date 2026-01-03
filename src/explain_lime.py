"""Generate a local explanation using LIME and save it as HTML."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import Config
from .utils import load_dataset, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/Crop_recommendation.csv")
    ap.add_argument("--sample_id", type=int, default=0, help="Row index from the dataset to explain (0-based).")
    ap.add_argument("--out", default="results/figures/lime_explanation.html")
    args = ap.parse_args()

    cfg = Config()
    X, y = load_dataset(args.data, cfg)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train a strong baseline model (Extra Trees)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y_enc
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            max_features=cfg.max_features,
        ))
    ])
    pipe.fit(X_train, y_train)

    # LIME explainer on training data (original feature space)
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(cfg.feature_cols),
        class_names=list(le.classes_),
        mode="classification",
        discretize_continuous=True,
        random_state=cfg.random_state,
    )

    sample_id = int(args.sample_id)
    if sample_id < 0 or sample_id >= len(X):
        raise ValueError(f"sample_id must be between 0 and {len(X)-1}")

    x0 = X.iloc[sample_id].values
    exp = explainer.explain_instance(x0, pipe.predict_proba, num_features=len(cfg.feature_cols))

    ensure_dir(Path(args.out).parent)
    exp.save_to_file(args.out)
    print("Saved LIME explanation to:", args.out)


if __name__ == "__main__":
    main()

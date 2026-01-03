"""Benchmark multiple classifiers and export metrics (Accuracy, Precision, Recall, F1)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import Config
from .utils import load_dataset, ensure_dir


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision_w": p, "recall_w": r, "f1_w": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/Crop_recommendation.csv")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--out", default="results/metrics/model_benchmark.csv")
    args = ap.parse_args()

    cfg = Config()
    X, y = load_dataset(args.data, cfg)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Models: keep scaling for models that need it; trees don't need it but can keep for consistency
    models = {
        "LogReg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        "SVM(RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=cfg.random_state),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=cfg.random_state, n_jobs=cfg.n_jobs),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=cfg.n_estimators, random_state=cfg.random_state, n_jobs=cfg.n_jobs),
        "GradientBoosting": GradientBoostingClassifier(random_state=cfg.random_state),
    }

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=cfg.random_state)

    rows = []
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y_enc, cv=skf)
        m = metrics(y_enc, y_pred)
        rows.append({"model": name, **m})

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    ensure_dir(Path(args.out).parent)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()

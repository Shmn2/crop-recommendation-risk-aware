import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: str, cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    missing = set(cfg.feature_cols + (cfg.label_col,)) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    X = df.loc[:, cfg.feature_cols].copy()
    y = df.loc[:, cfg.label_col].copy()
    return X, y


def stratified_split(X, y, test_size: float, random_state: int):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def save_json(obj, path: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) for multi-class classification using max-prob confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            bin_acc = acc[mask].mean()
            bin_conf = conf[mask].mean()
            ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

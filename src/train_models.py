"""Train and save the best model (Extra Trees) as a reproducible pipeline."""

import argparse
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import Config
from .utils import load_dataset, stratified_split, ensure_dir


def build_pipeline(cfg: Config, use_pca: bool = False, pca_components: int = 5):
    # For tree models scaling is not strictly required, but we keep it optional
    steps = []
    steps.append(("scaler", StandardScaler()))
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=cfg.random_state)))

    model = ExtraTreesClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        max_features=cfg.max_features,
    )
    steps.append(("model", model))
    return Pipeline(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/Crop_recommendation.csv")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--use_pca", action="store_true")
    ap.add_argument("--pca_components", type=int, default=5)
    args = ap.parse_args()

    cfg = Config()
    X, y = load_dataset(args.data, cfg)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = stratified_split(X, y_enc, cfg.test_size, cfg.random_state)

    pipe = build_pipeline(cfg, use_pca=args.use_pca, pca_components=args.pca_components)
    pipe.fit(X_train, y_train)

    ensure_dir(args.out_dir)
    joblib.dump(pipe, Path(args.out_dir) / "extra_trees_pipeline.joblib")
    joblib.dump(le, Path(args.out_dir) / "label_encoder.joblib")
    print("Saved models to:", args.out_dir)


if __name__ == "__main__":
    main()

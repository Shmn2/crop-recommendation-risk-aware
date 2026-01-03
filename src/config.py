from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data
    label_col: str = "label"
    feature_cols: tuple = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")

    # Reproducibility
    random_state: int = 42

    # Splits
    test_size: float = 0.30          # final test split
    calib_size: float = 0.20         # fraction of TRAIN used as conformal calibration

    # Model defaults
    n_estimators: int = 500
    max_features: str | float | None = "sqrt"
    n_jobs: int = -1

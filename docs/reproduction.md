# Reproducing Results

**Last updated:** 2026-01-03

## 1) Environment setup
```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## 2) Dataset
Kaggle dataset:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Place the CSV at:
`data/raw/Crop_recommendation.csv`

## 3) Run the pipeline
```bash
# Train and save the best model
python -m src.train_models --data data/raw/Crop_recommendation.csv

# Benchmark multiple models with 5-fold CV
python -m src.evaluate --data data/raw/Crop_recommendation.csv --cv 5

# Probability calibration (sigmoid or isotonic)
python -m src.calibrate --data data/raw/Crop_recommendation.csv --method isotonic

# Conformal prediction (alpha=0.1 -> ~90% target coverage)
python -m src.conformal --data data/raw/Crop_recommendation.csv --alpha 0.1

# LIME explanation (exports an HTML file)
python -m src.explain_lime --data data/raw/Crop_recommendation.csv --sample_id 0
```

## 4) Outputs
- Metrics: `results/metrics/`
- LIME HTML: `results/figures/lime_explanation.html`
- Saved models: `models/`

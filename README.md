# Risk-Aware and Explainable Crop Recommendation

This repository implements the methodology from the paper:
**Risk-Aware and Explainable Crop Recommendation Using Ensemble Learning, Conformal Prediction, and Probability Calibration**.

The system recommends crops from soil and environmental conditions:
**N, P, K, temperature, humidity, pH, rainfall**.

## Methodological contributions implemented
- **Ensemble learning** with Extra Trees (primary high-accuracy model)
- **Benchmarking** of multiple classifiers with stratified k-fold cross-validation
- **Probability calibration** (sigmoid and isotonic) + ECE reporting
- **Inductive conformal prediction** for risk-controlled recommendation sets
- **Explainability** using LIME (HTML export)

## Dataset
Kaggle Crop Recommendation Dataset:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset



## Setup
```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Reproduce results (recommended commands)
```bash
# Train + save Extra Trees pipeline
python -m src.train_models --data data/raw/Crop_recommendation.csv

# Benchmark models (5-fold CV)
python -m src.evaluate --data data/raw/Crop_recommendation.csv --cv 5

# Probability calibration + ECE
python -m src.calibrate --data data/raw/Crop_recommendation.csv --method isotonic

# Conformal prediction sets (alpha=0.1)
python -m src.conformal --data data/raw/Crop_recommendation.csv --alpha 0.1

# LIME explanation (HTML output)
python -m src.explain_lime --data data/raw/Crop_recommendation.csv --sample_id 0
```

## Outputs
- `results/metrics/` : CSV/JSON metrics (benchmarking, calibration, conformal)
- `results/figures/` : LIME explanation HTML
- `models/` : saved trained models

## Documentation
- `docs/methodology.md` : what is implemented and why
- `docs/reproduction.md` : step-by-step reproduction guide

## GitHub link (add after you publish)
After pushing this repo to GitHub, paste the link here and in your paper submission:
`https://github.com/<your-username>/crop-recommendation-risk-aware`

## Citation
If you use this repository, please cite the accompanying manuscript.

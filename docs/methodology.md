# Methodology (Implemented in this Repository)

This repository implements a **risk-aware and explainable crop recommendation** pipeline using the Crop Recommendation Dataset
(soil N, P, K and temperature, humidity, pH, rainfall; 22 crops).

## 1) Predictive Model (Ensemble Learning)
We use **Extra Trees** (Extremely Randomized Trees) as the primary model due to its strong performance for tabular data.
The training script builds a reproducible pipeline and saves model artifacts.

- Script: `src/train_models.py`
- Output: `models/extra_trees_pipeline.joblib`

## 2) Benchmarking and Metrics
Multiple classifiers are benchmarked using **stratified k-fold cross-validation**.
We report Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-score.

- Script: `src/evaluate.py`
- Output: `results/metrics/model_benchmark.csv`

## 3) Probability Calibration
To make confidence scores more reliable, we apply **sigmoid** and **isotonic** calibration using `CalibratedClassifierCV`.
We report Expected Calibration Error (ECE) and Log-loss before and after calibration.

- Script: `src/calibrate.py`
- Output: `results/metrics/calibration.json`

## 4) Conformal Prediction (Uncertainty-Aware Recommendation Sets)
We implement **inductive conformal prediction** to create prediction sets with approximately **(1 - alpha)** empirical coverage.
Using nonconformity scores `1 - P_hat(y|x)`, we compute a conformal threshold on a calibration set, then form a prediction set
for each test sample.

- Script: `src/conformal.py`
- Output: `results/metrics/conformal.json`

## 5) Explainability (LIME)
For local interpretability, we generate LIME explanations in the original agronomic feature space and export them as HTML.

- Script: `src/explain_lime.py`
- Output: `results/figures/lime_explanation.html`

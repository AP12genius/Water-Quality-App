# 💧 Water Quality Index (WQI) Forecaster & Analysis App

An interactive, machine-learning-powered web application that forecasts a river monitoring station's **next-year Water Quality Index (WQI)** from current-year physicochemical and biological observations.

Built with **Streamlit**, **Scikit-Learn**, and **Pandas**, featuring a leakage-safe chronological ML benchmark and a sleek, modern UI.

---

## 🚀 Live Demo & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## ✨ Features

- **Real-Time WQI Calculation & Forecasting**:
  - Automatically calculates current WQI from water sample measurements.
  - Predicts **next-year WQI** using a trained Random Forest Regressor.
  - Classifies water health status (🟢 *Excellent*, 🔵 *Good*, 🟡 *Medium*, 🟠 *Poor*, 🔴 *Very Poor*).
- **Interactive Parameter Sliders**:
  - Live adjustment of pH, Dissolved Oxygen, BOD, Coliform counts, and Nitrate.
  - Preset scenarios (Pristine Spring, Safe River, Urban Discharge, Severe Industrial Spill).
- **Safety & Usage Recommendations**:
  - Contextual advice for drinking, recreation, agricultural use, and ecological safety.
- **Model Benchmark & Diagnostics**:
  - Comparison across ML algorithms (Random Forest, XGBoost, LightGBM, SVR, Stacked Ensemble).
  - Visualization of actual vs. predicted performance, feature distributions, and partition metrics.

---

## 📊 Water Quality Parameters

The model evaluates standard surface water quality parameters:

| Parameter | Unit | Description | Optimal Range |
| :--- | :--- | :--- | :--- |
| **pH** | — | Acidity / alkalinity measure | 6.5 – 8.5 |
| **Dissolved Oxygen (DO)** | mg/L | Dissolved oxygen available for aquatic life | > 6.0 mg/L |
| **Biochemical Oxygen Demand (BOD)** | mg/L | Organic matter decomposition load | < 2.0 mg/L |
| **Total Coliform (TC)** | MPN/100 mL | General bacterial presence | < 500 MPN/100 mL |
| **Fecal Coliform (FC)** | MPN/100 mL | Human and animal waste contamination | < 100 MPN/100 mL |
| **Nitrate** | mg/L | Agricultural runoff / nutrient concentration | < 10 mg/L |

The target WQI combines these with documented weighted sub-indices: DO (0.31), FC (0.28), BOD (0.19), pH (0.11), and Nitrate (0.11).

---

## 🧠 Model Benchmark

To avoid target leakage, models are trained on strictly **consecutive station-year pairs** with a **chronological split**:
- **Training Set:** 2007–2012
- **Validation Set:** 2013
- **Final Test Set:** 2014

### Final Test Set Performance

| Model | R² Score | MAE | RMSE |
| :--- | :---: | :---: | :---: |
| **Persistence Baseline (Current WQI)** | 0.9076 | 3.14 | 4.33 |
| **Random Forest (Production Model)** | 0.8648 | 3.96 | 5.24 |
| **XGBoost** | 0.8562 | 3.95 | 5.40 |
| **LightGBM** | 0.8521 | 4.03 | 5.48 |
| **Stacked Ensemble** | 0.8554 | 4.00 | 5.42 |
| **SVR** | 0.7517 | 5.05 | 7.10 |

---

## 📁 Repository Structure

```
Water-Quality-App/
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Application dependencies for local run & cloud deployment
├── .streamlit/
│   └── config.toml             # Custom styling and theme settings
├── models/
│   ├── rf.joblib               # Serialized Random Forest model
│   └── preprocessor.joblib     # Preprocessing pipeline
├── data/
│   └── dataset.csv             # Historical water quality monitoring dataset
├── assets/                     # Diagnostic and model performance visual plots
├── src/
│   ├── model.py                # Data processing, feature engineering & model pipelines
│   └── hybrid_model.py         # Stacking ensemble and benchmark routines
├── main.ipynb                  # Exploratory analysis, training, and SHAP interpretability
├── save_models.py              # Script to train and re-export model artifacts
└── README.md
```

---

## ☁️ Deployment on Streamlit Community Cloud

This repository is configured for 1-click deployment on **Streamlit Community Cloud**:

1. Fork or clone this repository to your GitHub account.
2. Sign in to **[share.streamlit.io](https://share.streamlit.io)** with GitHub.
3. Select this repository, set Branch to `main`, and Main file path to `app.py`.
4. Click **Deploy!**.

---

## 📖 Methodology & Reference

This project builds on methodologies and benchmarks established in water informatics research:

> Lamba, S., Dawar, I., Singal, M., & Singh, J. (2025). *Predicting water quality index using machine learning techniques: a case study of river Ganga in Haridwar, India.* Earth Science Informatics, 18, 369. [DOI: 10.1007/s12145-025-01865-2](https://doi.org/10.1007/s12145-025-01865-2)

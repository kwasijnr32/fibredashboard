# AI-Based Predictive Fault Detection in Optical Fiber Networks
**Final Year Undergraduate Project | 2025–2026**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full training pipeline (RF + LSTM)
python train_evaluate.py

# RF only (faster, no TensorFlow needed)
python train_evaluate.py --no-lstm

# Launch monitoring dashboard
streamlit run app.py
```

## Project Structure

```
fiber_project/
├── data/
│   └── generate_data.py       ← Synthetic fiber degradation dataset generator
├── utils/
│   ├── preprocessing.py       ← Feature engineering, health index, risk labels
│   └── metrics.py             ← MAE, RMSE, R², precision/recall/F1
├── models/
│   ├── rf_model.py            ← Random Forest with bootstrap confidence intervals
│   └── lstm_model.py          ← Bidirectional LSTM with MC Dropout uncertainty
├── dashboard/
│   └── app.py                 ← Streamlit real-time monitoring dashboard
├── train_evaluate.py          ← Main pipeline: generate → train → evaluate
├── requirements.txt
└── README.md
```

## System Overview

The system monitors two telemetry signals per fiber link:
- **Attenuation A(t)** — signal loss in dB/km
- **PMD D(t)** — Polarization Mode Dispersion in ps/√km

From these, it computes a **Health Index H(t)** and classifies links as:

| H(t) | Risk Class | Action |
|------|-----------|--------|
| < 0.40 | 🟢 Low | Normal operations |
| 0.40–0.70 | 🟡 Medium | Increase monitoring |
| ≥ 0.70 | 🔴 High | Immediate inspection |

## Models

| Model | Approach | Uncertainty |
|-------|----------|-------------|
| Random Forest | Tabular features, per-step | Tree quantile intervals |
| Bidirectional LSTM | Sequences (lookback=24) | Monte Carlo Dropout |

## Outputs

After running `train_evaluate.py`:
- `outputs/fiber_dataset.csv` — generated dataset
- `outputs/fiber_features.csv` — engineered features
- `outputs/rf_model.joblib` — saved RF model
- `outputs/rf_predictions.png` — forecast plots
- `outputs/rf_importance_*.png` — feature importance charts
- `outputs/lstm_model.keras` — saved LSTM (if TF available)
- `outputs/lstm_training.png` — training curves

## Dashboard

The Streamlit dashboard provides:
- Per-link KPI cards (risk status, current A and D values)
- Time-series plots with forecast curves and confidence intervals
- Health index gauge with risk alert
- Risk class distribution over time
- Feature importance charts

## Timeline (8 Weeks)

1. **Week 1–2**: Dataset design, generator, preprocessing ← *You are here*
2. **Week 3–4**: Baseline ML models and evaluation
3. **Week 5**: LSTM + uncertainty quantification
4. **Week 6**: Dashboard prototype
5. **Week 7**: Ablation experiments & hyperparameter tuning
6. **Week 8**: Report writing & presentation

## Dependencies

- Python ≥ 3.9
- numpy, pandas, scikit-learn, matplotlib, plotly, streamlit, joblib
- tensorflow ≥ 2.13 (optional, for LSTM)

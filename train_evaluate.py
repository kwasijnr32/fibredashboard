"""
Main Training & Evaluation Pipeline
Runs full experiment: data generation → preprocessing → RF + LSTM training → evaluation
Usage: python train_evaluate.py [--tune] [--no-lstm]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate_fiber_dataset
from utils.preprocessing import (
    clean_series, add_features, compute_health_index,
    make_sequences, time_split, THETA_1, THETA_2
)
from utils.metrics import (
    regression_metrics, classification_metrics,
    print_regression_metrics, print_classification_metrics
)
from models.rf_model import FiberRFModel


def assign_risk(health_vals):
    return pd.cut(
        health_vals,
        bins=[-np.inf, THETA_1, THETA_2, np.inf],
        labels=["Low", "Medium", "High"]
    )


def run_pipeline(tune_rf: bool = False, use_lstm: bool = True,
                 T: int = 500, n_links: int = 5,
                 lookback: int = 24, horizon: int = 6):

    os.makedirs("outputs", exist_ok=True)

    # ── 1. Generate & preprocess data ─────────────────────────────────────
    print("\n[1/5] Generating synthetic dataset...")
    raw = generate_fiber_dataset(T=T, n_links=n_links)
    raw.to_csv("outputs/fiber_dataset.csv", index=False)

    print("[2/5] Preprocessing & feature engineering...")
    clean = clean_series(raw)
    feat = add_features(clean)
    final = compute_health_index(feat)
    final.to_csv("outputs/fiber_features.csv", index=False)

    print(f"  Dataset shape: {final.shape}")
    print(f"  Risk distribution:\n{final['risk_class'].value_counts()}")

    # ── Feature & target columns ──────────────────────────────────────────
    exclude = {"timestamp", "link_id", "attenuation_dB_km", "pmd_ps_sqkm",
               "health_index", "risk_class", "risk_numeric"}
    feature_cols = [c for c in final.columns if c not in exclude]
    target_cols = ["attenuation_dB_km", "pmd_ps_sqkm"]

    # ── 2. Time split ──────────────────────────────────────────────────────
    print("[3/5] Splitting data (train / val / test)...")
    train_df, val_df, test_df = time_split(final)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── 3a. Random Forest Pipeline ─────────────────────────────────────────
    print("\n[4/5] Training Random Forest model...")

    # For RF: create horizon-shifted targets
    def build_rf_dataset(df):
        df = df.sort_values(["link_id", "timestamp"])
        shifted_targets = df.groupby("link_id")[target_cols].shift(-horizon)
        df = pd.concat([df, shifted_targets.add_suffix("_target")], axis=1).dropna()
        X = df[feature_cols].values
        y = df[[f"{c}_target" for c in target_cols]].values
        return X, y, df

    X_tr, y_tr, _ = build_rf_dataset(train_df)
    X_val, y_val, _ = build_rf_dataset(val_df)
    X_te, y_te, test_with_tgt = build_rf_dataset(test_df)

    rf = FiberRFModel(n_estimators=200)
    rf.fit(X_tr, y_tr, feature_cols=feature_cols, tune=tune_rf)
    rf.save("outputs/rf_model.joblib")

    rf_preds = rf.predict(X_te)
    print_regression_metrics(regression_metrics(y_te[:, 0], rf_preds[:, 0]),
                              "RF", "Attenuation")
    print_regression_metrics(regression_metrics(y_te[:, 1], rf_preds[:, 1]),
                              "RF", "PMD")

    # Risk classification from RF predictions
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    health_true = 0.6 * scaler.fit_transform(y_te[:, 0:1]).flatten() + \
                  0.4 * scaler.fit_transform(y_te[:, 1:2]).flatten()
    health_pred = 0.6 * scaler.fit_transform(rf_preds[:, 0:1]).flatten() + \
                  0.4 * scaler.fit_transform(rf_preds[:, 1:2]).flatten()

    risk_true = assign_risk(health_true)
    risk_pred = assign_risk(health_pred)
    print_classification_metrics(
        classification_metrics(risk_true, risk_pred), "RF"
    )

    # Feature importance plot
    fi = rf.feature_importance()
    for tgt in target_cols:
        sub = fi[fi["target"] == tgt].head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(sub["feature"], sub["importance"], color="#2196F3")
        ax.set_title(f"RF Feature Importance — {tgt}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"outputs/rf_importance_{tgt.split('_')[0]}.png", dpi=150)
        plt.close()

    # ── 3b. RF prediction plot ────────────────────────────────────────────
    plot_predictions(y_te, rf_preds, title="RF Forecasts", suffix="rf")

    # ── 4. LSTM Pipeline ──────────────────────────────────────────────────
    if use_lstm:
        try:
            from models.lstm_model import FiberLSTMModel, TF_AVAILABLE
            if not TF_AVAILABLE:
                print("\n⚠️  Skipping LSTM (TensorFlow not available).")
            else:
                print("\n[5/5] Training LSTM model...")
                X_seq_tr, y_seq_tr = make_sequences(
                    train_df, feature_cols, target_cols, lookback, horizon)
                X_seq_val, y_seq_val = make_sequences(
                    val_df, feature_cols, target_cols, lookback, horizon)
                X_seq_te, y_seq_te = make_sequences(
                    test_df, feature_cols, target_cols, lookback, horizon)

                # Normalize features for LSTM
                sc = StandardScaler()
                shape = X_seq_tr.shape
                X_seq_tr_sc = sc.fit_transform(X_seq_tr.reshape(-1, shape[-1])).reshape(shape)
                X_seq_val_sc = sc.transform(X_seq_val.reshape(-1, shape[-1])).reshape(X_seq_val.shape)
                X_seq_te_sc = sc.transform(X_seq_te.reshape(-1, shape[-1])).reshape(X_seq_te.shape)

                lstm = FiberLSTMModel(lookback=lookback, n_features=shape[-1])
                lstm.fit(X_seq_tr_sc, y_seq_tr, X_seq_val_sc, y_seq_val, epochs=80)
                lstm.save("outputs/lstm_model.keras")
                lstm.plot_training("outputs/lstm_training.png")

                lstm_preds = lstm.predict(X_seq_te_sc)
                print_regression_metrics(
                    regression_metrics(y_seq_te[:, 0], lstm_preds[:, 0]),
                    "LSTM", "Attenuation")
                print_regression_metrics(
                    regression_metrics(y_seq_te[:, 1], lstm_preds[:, 1]),
                    "LSTM", "PMD")

                plot_predictions(y_seq_te, lstm_preds, title="LSTM Forecasts", suffix="lstm")
        except Exception as e:
            print(f"\n⚠️  LSTM training failed: {e}")

    print("\n✅  Pipeline complete. Outputs saved to outputs/")


def plot_predictions(y_true, y_pred, title="Forecasts", suffix="model"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    labels = ["Attenuation (dB/km)", "PMD (ps/√km)"]
    for i, ax in enumerate(axes):
        n = min(200, len(y_true))
        ax.plot(y_true[:n, i], label="Actual", color="#1976D2", lw=1.5)
        ax.plot(y_pred[:n, i], label="Predicted", color="#E53935", lw=1.5, linestyle="--")
        ax.set_title(f"{title} — {labels[i]}")
        ax.set_xlabel("Sample")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{suffix}_predictions.png", dpi=150)
    plt.close()
    print(f"  Prediction plot saved → outputs/{suffix}_predictions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run RF hyperparameter tuning")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    args = parser.parse_args()

    run_pipeline(tune_rf=args.tune, use_lstm=not args.no_lstm)

"""
Preprocessing and Feature Engineering for Fiber Degradation Data
- Handles missing values & outliers
- Creates rolling, lag, slope features
- Computes normalized health index H(t) and risk classes
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


# ── Risk thresholds (percentile-based on health index) ──────────────────────
THETA_1 = 0.40   # Low → Medium boundary
THETA_2 = 0.70   # Medium → High boundary

# ── Health index weights ─────────────────────────────────────────────────────
W_A = 0.6   # attenuation weight
W_D = 0.4   # PMD weight


def clean_series(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and clip outliers using IQR."""
    df = df.copy()
    for col in ["attenuation_dB_km", "pmd_ps_sqkm"]:
        # Forward-fill then back-fill NaNs
        df[col] = df[col].ffill().bfill()
        # IQR-based outlier clipping
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)
    return df


def add_features(df: pd.DataFrame, windows: list = [6, 12, 24]) -> pd.DataFrame:
    """
    Add rolling statistics and lag features per link.
    windows: rolling window sizes (number of steps, each step = 6h)
    """
    df = df.copy().sort_values(["link_id", "timestamp"])
    new_cols = {}

    for col in ["attenuation_dB_km", "pmd_ps_sqkm"]:
        for w in windows:
            new_cols[f"{col}_roll_mean_{w}"] = (
                df.groupby("link_id")[col].transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            new_cols[f"{col}_roll_std_{w}"] = (
                df.groupby("link_id")[col].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            )
            # Rolling slope (linear trend over window)
            def rolling_slope(x, w=w):
                def slope(vals):
                    if len(vals) < 2:
                        return 0.0
                    t = np.arange(len(vals))
                    return np.polyfit(t, vals, 1)[0]
                return x.rolling(w, min_periods=2).apply(slope, raw=True).fillna(0)

            new_cols[f"{col}_slope_{w}"] = (
                df.groupby("link_id")[col].transform(rolling_slope)
            )

        # Lag features
        for lag in [1, 3, 6]:
            new_cols[f"{col}_lag_{lag}"] = (
                df.groupby("link_id")[col].shift(lag)
            )

        # Change rate
        new_cols[f"{col}_change"] = (
            df.groupby("link_id")[col].diff().fillna(0)
        )

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    df = df.dropna().reset_index(drop=True)
    return df


def compute_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized health index H(t) and risk class R(t).
    H(t) = wA * A_norm(t) + wD * D_norm(t)
    """
    df = df.copy()
    scaler = MinMaxScaler()
    A_norm = scaler.fit_transform(df[["attenuation_dB_km"]]).flatten()
    D_norm = scaler.fit_transform(df[["pmd_ps_sqkm"]]).flatten()

    df["health_index"] = W_A * A_norm + W_D * D_norm

    df["risk_class"] = pd.cut(
        df["health_index"],
        bins=[-np.inf, THETA_1, THETA_2, np.inf],
        labels=["Low", "Medium", "High"]
    )
    df["risk_numeric"] = df["risk_class"].map({"Low": 0, "Medium": 1, "High": 2})
    return df


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    lookback: int = 24,
    horizon: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays for sequence-to-point regression.
    X shape: (N, lookback, n_features)
    y shape: (N, len(target_cols))
    """
    X_list, y_list = [], []
    for link_id, grp in df.groupby("link_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        feat = grp[feature_cols].values
        tgt = grp[target_cols].values
        for i in range(lookback, len(grp) - horizon + 1):
            X_list.append(feat[i - lookback:i])
            y_list.append(tgt[i + horizon - 1])
    return np.array(X_list), np.array(y_list)


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time (no data leakage)."""
    df = df.sort_values("timestamp")
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


if __name__ == "__main__":
    from data.generate_data import generate_fiber_dataset

    raw = generate_fiber_dataset(T=300, n_links=3)
    clean = clean_series(raw)
    feat = add_features(clean)
    final = compute_health_index(feat)
    print(final[["timestamp", "link_id", "attenuation_dB_km", "pmd_ps_sqkm",
                  "health_index", "risk_class"]].head(20))
    print("\nRisk class distribution:")
    print(final["risk_class"].value_counts())

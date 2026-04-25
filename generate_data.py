"""
Synthetic Dataset Generator for Optical Fiber Degradation
Generates realistic attenuation (dB/km) and PMD (ps/sqrt(km)) time-series
with degradation trends, stress events, and noise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)


def generate_stress_events(T: int, n_events: int = 5) -> np.ndarray:
    """Generate random stress spike events (bending/pressure bursts)."""
    events = np.zeros(T)
    event_times = np.random.choice(T, size=n_events, replace=False)
    for t in event_times:
        duration = np.random.randint(3, 15)
        magnitude = np.random.uniform(0.05, 0.25)
        decay = np.exp(-np.arange(duration) / (duration / 3))
        end = min(t + duration, T)
        events[t:end] += magnitude * decay[:end - t]
    return events


def generate_fiber_dataset(
    T: int = 500,
    n_links: int = 5,
    start_date: str = "2023-01-01",
    A0_range: tuple = (0.18, 0.22),
    D0_range: tuple = (0.05, 0.12),
    alpha_range: tuple = (1e-5, 5e-5),   # attenuation drift rate per time step
    beta_range: tuple = (5e-6, 2e-5),    # PMD drift rate per time step
    noise_A: float = 0.003,
    noise_D: float = 0.002,
) -> pd.DataFrame:
    """
    Generate synthetic fiber degradation dataset.

    Returns a DataFrame with columns:
        timestamp, link_id, attenuation_dB_km, pmd_ps_sqkm
    """
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [start + timedelta(hours=i * 6) for i in range(T)]  # every 6h

    for link_id in range(1, n_links + 1):
        A0 = np.random.uniform(*A0_range)
        D0 = np.random.uniform(*D0_range)
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)

        t_arr = np.arange(T)

        # Base degradation trends
        A_trend = A0 + alpha * t_arr
        D_trend = D0 + beta * t_arr

        # Noise
        eps_A = np.random.normal(0, noise_A, T)
        eps_D = np.random.normal(0, noise_D, T)

        # Stress events
        stress_A = generate_stress_events(T, n_events=np.random.randint(3, 8))
        stress_D = generate_stress_events(T, n_events=np.random.randint(3, 8)) * 0.5

        A = A_trend + eps_A + stress_A
        D = D_trend + eps_D + stress_D

        # Clip to physical range
        A = np.clip(A, 0.15, 0.60)
        D = np.clip(D, 0.02, 0.50)

        for i in range(T):
            records.append({
                "timestamp": timestamps[i],
                "link_id": f"Link_{link_id:02d}",
                "attenuation_dB_km": round(A[i], 5),
                "pmd_ps_sqkm": round(D[i], 5),
            })

    df = pd.DataFrame(records).sort_values(["link_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    df = generate_fiber_dataset(T=500, n_links=5)
    out_path = "outputs/fiber_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset saved → {out_path}")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(df.describe())

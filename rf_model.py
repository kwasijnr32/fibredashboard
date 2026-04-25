"""
Random Forest Regression Model for Fiber Degradation Forecasting
Predicts future attenuation and PMD values from engineered features.
Includes bootstrap-based confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import os
from typing import Tuple, Dict


class FiberRFModel:
    """
    Random Forest baseline for attenuation + PMD forecasting.

    The model treats each time step independently (non-sequential).
    Features: rolling stats, lags, slopes from preprocessing pipeline.
    Targets: [attenuation_dB_km, pmd_ps_sqkm] at t+horizon
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = None,
                 n_jobs: int = -1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        self.models: Dict[str, RandomForestRegressor] = {}
        self.feature_cols = None
        self.target_cols = ["attenuation_dB_km", "pmd_ps_sqkm"]
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_cols: list, tune: bool = False):
        """
        X_train: (N, n_features) flat feature matrix
        y_train: (N, 2) — [attenuation, pmd]
        """
        self.feature_cols = feature_cols
        X_scaled = self.scaler.fit_transform(X_train)

        for i, col in enumerate(self.target_cols):
            print(f"  Training RF for {col}...")
            if tune:
                param_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 2, 5],
                }
                base = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
                gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=self.n_jobs)
                gs.fit(X_scaled, y_train[:, i])
                self.models[col] = gs.best_estimator_
                print(f"    Best params: {gs.best_params_}")
            else:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                rf.fit(X_scaled, y_train[:, i])
                self.models[col] = rf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions of shape (N, 2)."""
        X_scaled = self.scaler.transform(X)
        preds = np.column_stack([self.models[col].predict(X_scaled)
                                  for col in self.target_cols])
        return preds

    def predict_with_intervals(self, X: np.ndarray,
                                alpha: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Bootstrap confidence intervals using individual tree predictions.
        Returns dict with keys: mean, lower, upper for each target.
        """
        X_scaled = self.scaler.transform(X)
        results = {}
        for col in self.target_cols:
            rf = self.models[col]
            tree_preds = np.array([tree.predict(X_scaled) for tree in rf.estimators_])
            results[col] = {
                "mean": tree_preds.mean(axis=0),
                "lower": np.percentile(tree_preds, 100 * alpha / 2, axis=0),
                "upper": np.percentile(tree_preds, 100 * (1 - alpha / 2), axis=0),
            }
        return results

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances for each target."""
        rows = []
        for col in self.target_cols:
            rf = self.models[col]
            for feat, imp in zip(self.feature_cols, rf.feature_importances_):
                rows.append({"target": col, "feature": feat, "importance": imp})
        return pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False])

    def save(self, path: str = "outputs/rf_model.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"RF model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FiberRFModel":
        return joblib.load(path)

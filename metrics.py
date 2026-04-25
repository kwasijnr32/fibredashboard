"""
Evaluation Metrics for Fiber Degradation Forecasting
Implements MAE, RMSE, R², and classification metrics (precision/recall/F1)
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from typing import Dict


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R² for regression targets."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 6), "RMSE": round(rmse, 6), "R2": round(r2, 4)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            labels: list = ["Low", "Medium", "High"]) -> Dict:
    """Compute precision, recall, F1 per risk class."""
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"report": report, "confusion_matrix": cm}


def print_regression_metrics(metrics: Dict, model_name: str = "", target: str = ""):
    label = f"[{model_name}] {target}" if model_name else target
    print(f"\n── Regression Metrics {label} ──")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v}")


def print_classification_metrics(metrics: Dict, model_name: str = ""):
    print(f"\n── Classification Metrics [{model_name}] ──")
    report = metrics["report"]
    for cls in ["Low", "Medium", "High"]:
        if cls in report:
            r = report[cls]
            print(f"  {cls:8s} | P={r['precision']:.3f} | R={r['recall']:.3f} | F1={r['f1-score']:.3f} | support={int(r['support'])}")
    print(f"\n  Confusion Matrix:\n{metrics['confusion_matrix']}")

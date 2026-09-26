from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Compute classification metrics for binary spam detection."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> np.ndarray:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return confusion_matrix(y_true, y_pred, labels=[1, 0])


def format_metric(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_model_comparison_table(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    comparison_rows: List[Dict[str, object]] = []
    for result in results:
        comparison_rows.append(
            {
                "Model": result["model_name"],
                "Vectorizer": result["vectorizer_name"],
                "Accuracy": float(result["accuracy"]),
                "Precision": float(result["precision"]),
                "Recall": float(result["recall"]),
                "F1": float(result["f1"]),
            }
        )
    return comparison_rows

"""Cálculo de métricas de evaluación (fase 05 - evaluación)."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import resolve_path
from src.models.base import Evaluator


class ClassificationEvaluator(Evaluator):
    """Calcula el set de métricas de clasificación binaria acordado en
    `01_comprension_del_negocio.ipynb` / README (`Resultados principales`).
    """

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

    def confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)

    def roc_curve_points(self, y_true: pd.Series, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        return fpr, tpr


def save_metrics_table(metrics_by_model: dict[str, dict], config: dict, filename: str = "metricas_modelos.csv") -> None:
    """Guarda una tabla comparativa de métricas por modelo en
    `reports/tables/`, para alimentar la sección 'Resultados principales'
    del README.
    """
    tables_dir = resolve_path(config["paths"]["reports"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_by_model).T
    df.index.name = "modelo"
    df.to_csv(tables_dir / filename)


def save_metrics_json(metrics: dict, config: dict, filename: str) -> None:
    tables_dir = resolve_path(config["paths"]["reports"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / filename).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

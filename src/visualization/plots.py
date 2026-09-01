"""Funciones de visualización reutilizables.

Cubren los TODOs dejados pendientes en `02_comprension_de_los_datos.ipynb`
(distribuciones, nulos, correlación) y los gráficos de evaluación de
`05_evaluacion.ipynb` / `06_interpretacion_y_resultados.ipynb` (matriz de
confusión, curva ROC, importancia de variables). Cada función guarda su
figura en `reports/figures/` (ruta resuelta vía `config.yaml`) y también la
devuelve, para poder mostrarla inline en el notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import resolve_path


def _figures_dir(config: dict) -> Path:
    figures_dir = resolve_path(config["paths"]["reports"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _save(fig: plt.Figure, config: dict, filename: str) -> Path:
    path = _figures_dir(config) / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path


# --- Fase 02: comprensión de los datos -------------------------------------------


def plot_score_distribution(df: pd.DataFrame, column: str, config: dict, filename: str) -> plt.Figure:
    """Histograma + boxplot de un puntaje de catación (ej. Total.Cup.Points)."""
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df[column].dropna(), kde=True, ax=ax_hist)
    ax_hist.set_title(f"Distribución de {column}")
    sns.boxplot(x=df[column].dropna(), ax=ax_box)
    ax_box.set_title(f"Boxplot de {column}")
    fig.tight_layout()
    _save(fig, config, filename)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], config: dict, filename: str) -> plt.Figure:
    """Matriz de correlación entre columnas numéricas (ej. subpuntajes SCA)."""
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    _save(fig, config, filename)
    return fig


def plot_categorical_counts(
    df: pd.DataFrame, column: str, config: dict, filename: str, top_n: int = 15
) -> plt.Figure:
    """Barras con las categorías más frecuentes de una variable categórica."""
    counts = df[column].value_counts(dropna=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, orient="h")
    ax.set_xlabel("Cantidad de lotes")
    ax.set_title(f"Top {top_n} categorías de {column}")
    fig.tight_layout()
    _save(fig, config, filename)
    return fig


# --- Fase 05: evaluación -----------------------------------------------------------


def plot_confusion_matrix(cm: np.ndarray, config: dict, filename: str, labels: tuple[str, str] = ("No especialidad", "Especialidad")) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    _save(fig, config, filename)
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, config: dict, filename: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC")
    ax.legend()
    fig.tight_layout()
    _save(fig, config, filename)
    return fig


# --- Fase 06: interpretación --------------------------------------------------------


def plot_logistic_coefficients(
    feature_names: list[str], coefficients: np.ndarray, config: dict, filename: str, top_n: int = 20
) -> plt.Figure:
    """Grafica los coeficientes de mayor magnitud de la regresión logística,
    como aproximación de qué variables se asocian más con 'especialidad'.
    """
    # Los nombres vienen con el prefijo del ColumnTransformer (ej.
    # "categorical__Country.of.Origin_Kenya"); se quita para que el gráfico
    # sea legible para una audiencia de negocio.
    clean_names = [name.split("__", 1)[-1] for name in feature_names]
    coef_series = pd.Series(coefficients, index=clean_names)
    top_coefs = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in top_coefs.values]
    ax.barh(top_coefs.index[::-1], top_coefs.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente (log-odds)")
    ax.set_title("Variables más asociadas con 'café de especialidad'")
    fig.tight_layout()
    _save(fig, config, filename)
    return fig

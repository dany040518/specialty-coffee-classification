"""Funciones de visualización reutilizables (EDA de las fases 02 y 03).

Distribuciones, mapas de nulos, matrices de correlación y grillas de
boxplots/histogramas. Cada función guarda su figura en `reports/figures/`
(ruta resuelta vía `config.yaml`) y también la devuelve, para mostrarla
inline en el notebook.
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


def plot_missing_matrix(
    df: pd.DataFrame, config: dict, filename: str, title: str = "Mapa de valores faltantes"
) -> plt.Figure:
    """Mapa de calor de valores faltantes (una franja por columna): cada celda
    marcada = valor nulo. Permite ver de un vistazo qué columnas concentran los
    nulos y si faltan "en bloque" (mismas filas) o de forma dispersa.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Columnas")
    fig.tight_layout()
    _save(fig, config, filename)
    return fig


def plot_feature_distributions(
    df: pd.DataFrame, columns: list[str], config: dict, filename: str, ncols: int = 4, bins: int = 30
) -> plt.Figure:
    """Grilla de histogramas (con KDE) para varias variables numéricas a la vez,
    por ejemplo los subpuntajes de catación que alimentan el clustering.
    """
    cols = [c for c in columns if c in df.columns]
    nrows = -(-len(cols) // ncols)  # techo de la división
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, bins=bins, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, config, filename)
    return fig


def plot_boxplots(
    df: pd.DataFrame, columns: list[str], config: dict, filename: str, ncols: int = 5
) -> plt.Figure:
    """Grilla de boxplots para inspeccionar valores atípicos y variables casi
    constantes en un conjunto de columnas numéricas.
    """
    cols = [c for c in columns if c in df.columns]
    nrows = -(-len(cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col].dropna(), ax=ax)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, config, filename)
    return fig

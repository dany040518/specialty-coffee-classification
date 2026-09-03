"""Interpretación y perfilado de los grupos (fase 06 - interpretación).

Funciones planas que reproducen lo que hace `notebooks/06_interpretacion_y_resultados.ipynb`:

- `centers_in_points`: revierte la estandarización de la fase 03 para expresar el
  centro de cada grupo en puntos SCA en vez de en desviaciones (sección 6.4),
- `profile_categorical`: reparto de una variable categórica dentro de cada grupo,
  en conteo y en porcentaje (sección 6.5),
- `profile_numeric`: estadísticas de una variable numérica por grupo, calculadas
  solo con los lotes que reportan el dato (sección 6.6),
- `profile_reporting`: qué porcentaje de cada grupo reporta cada dato, a partir de
  las banderas que dejó la fase 03 (sección 6.7),
- `build_dashboard_dataset`: el dataset plano que consume el tablero (sección 6.12).

Los porcentajes de `profile_categorical` son siempre **dentro del grupo**, no
sobre el total, porque los grupos tienen tamaños muy distintos (708, 295 y 307
lotes) y comparar conteos daría una lectura falsa.

Los nombres de los grupos no se calculan: son una decisión editorial que vive en
`config -> interpretation.group_names`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def centers_in_points(
    X: pd.DataFrame, labels: np.ndarray, scaler_params: dict
) -> pd.DataFrame:
    """Devuelve el centro de cada grupo en puntos SCA (0 a 10).

    El agrupamiento trabajó con los atributos estandarizados, así que sus centros
    quedan en desviaciones respecto al promedio y no significan nada fuera del
    modelo. Con la media y la escala que guardó la fase 03 se revierte la
    transformación: `valor_original = valor_estandarizado * escala + media`.

    Args:
        X: matriz de entrada estandarizada (`clustering_input.csv`).
        labels: etiqueta de grupo de cada lote.
        scaler_params: contenido de `data/processed/clustering_scaler.json`.

    Returns:
        DataFrame con una fila por grupo y una columna por atributo de catación.

    Raises:
        ValueError: si el orden de las columnas de `X` no es el del scaler, porque
            la reversión se hace posición por posición.
    """
    if scaler_params["features"] != list(X.columns):
        raise ValueError("el orden de las columnas no coincide con el del scaler")
    media = np.array(scaler_params["mean"])
    escala = np.array(scaler_params["scale"])
    return X.groupby(labels).mean() * escala + media


def profile_categorical(
    df: pd.DataFrame, column: str, group_column: str = "grupo"
) -> pd.DataFrame:
    """Reparto de una variable categórica dentro de cada grupo.

    Args:
        df: dataset con la columna de grupo y la variable a perfilar.
        column: variable categórica (país, variedad, procesamiento, color).
        group_column: columna con el grupo de cada lote.

    Returns:
        DataFrame en formato largo con `grupo`, `valor`, `n` y
        `pct_dentro_del_grupo`, ordenado por grupo y frecuencia.
    """
    rows = []
    for group in sorted(df[group_column].unique()):
        subset = df[df[group_column] == group]
        for value, n in subset[column].value_counts().items():
            rows.append(
                {
                    "grupo": group,
                    "valor": value,
                    "n": int(n),
                    "pct_dentro_del_grupo": round(n / len(subset) * 100, 1),
                }
            )
    return pd.DataFrame(rows)


def profile_numeric(
    df: pd.DataFrame, columns: list[str], group_column: str = "grupo"
) -> pd.DataFrame:
    """Estadísticas de varias variables numéricas por grupo.

    Solo entran los lotes que reportan el dato: la fase 03 decidió no imputar
    `altitude_mean_meters` ni `Moisture` porque sus faltantes no son aleatorios.
    Por eso la tabla trae `n_reporta` y `pct_reporta`, para leer cada promedio
    sabiendo sobre cuántos lotes se calculó.

    Args:
        df: dataset con la columna de grupo.
        columns: variables numéricas de perfilado.
        group_column: columna con el grupo de cada lote.

    Returns:
        DataFrame en formato largo con una fila por variable y grupo.
    """
    rows = []
    for column in columns:
        for group in sorted(df[group_column].unique()):
            values = df.loc[df[group_column] == group, column].dropna()
            n_group = int((df[group_column] == group).sum())
            rows.append(
                {
                    "variable": column,
                    "grupo": group,
                    "n_reporta": len(values),
                    "n_grupo": n_group,
                    "pct_reporta": round(len(values) / n_group * 100, 1),
                    "media": round(values.mean(), 2),
                    "desviacion": round(values.std(), 2),
                    "minimo": round(values.min(), 2),
                    "maximo": round(values.max(), 2),
                }
            )
    return pd.DataFrame(rows)


def profile_defects(
    df: pd.DataFrame, columns: list[str], group_column: str = "grupo"
) -> pd.DataFrame:
    """Defectos por grupo: promedio, mediana, extremos y proporción de lotes afectados.

    Se reportan las dos cosas porque son preguntas distintas: cuántos defectos
    trae un lote típico (la mediana es 0 en casi todos los grupos) y qué
    proporción de lotes trae al menos uno.
    """
    rows = []
    for column in columns:
        for group in sorted(df[group_column].unique()):
            values = df.loc[df[group_column] == group, column].dropna()
            rows.append(
                {
                    "variable": column,
                    "grupo": group,
                    "n": len(values),
                    "media": round(values.mean(), 2),
                    "mediana": round(values.median(), 2),
                    "minimo": round(values.min(), 2),
                    "maximo": round(values.max(), 2),
                    "pct_con_al_menos_uno": round((values > 0).mean() * 100, 1),
                }
            )
    return pd.DataFrame(rows)


def profile_reporting(
    df: pd.DataFrame, flags: list[str], group_column: str = "grupo"
) -> pd.DataFrame:
    """Qué porcentaje de cada grupo reporta cada dato.

    Usa las banderas `altitude_reportada` y `humedad_reportada` que creó la fase
    03. Un grupo que reporta menos no tiene peores cafés: tiene otros países
    adentro, y son los países los que llenan (o no) el formulario.
    """
    rows = []
    for group in sorted(df[group_column].unique()):
        subset = df[df[group_column] == group]
        row = {"grupo": group, "n_grupo": len(subset)}
        for flag in flags:
            row[f"{flag}_si_pct"] = round(subset[flag].mean() * 100, 1)
            row[f"{flag}_no_pct"] = round((1 - subset[flag].mean()) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_dashboard_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Arma el dataset plano que consume el tablero.

    Agrega tres columnas a `coffee_clustered.csv`: `especialidad` (1 si el lote
    supera `validation.specialty_threshold`), `grupo_nombre` (el nombre en
    palabras de `interpretation.group_names`) y `rango_puntaje` (cuatro tramos
    anclados en el umbral, de `interpretation.score_band_width` puntos de ancho).

    Args:
        df: dataset con la columna `grupo` (`coffee_clustered.csv`).
        config: contenido de `config/config.yaml`.

    Returns:
        Una copia de `df` con las tres columnas agregadas.
    """
    reference_column = config["validation"]["reference_column"]
    threshold = config["validation"]["specialty_threshold"]
    group_names = config["interpretation"]["group_names"]
    width = config["interpretation"]["score_band_width"]

    dashboard = df.copy()
    dashboard["especialidad"] = (dashboard[reference_column] >= threshold).astype(int)
    dashboard["grupo_nombre"] = dashboard["grupo"].map(group_names)

    cuts = [-np.inf, threshold - width, threshold, threshold + width, np.inf]
    labels = [
        f"menos de {threshold - width}",
        f"{threshold - width} a {threshold}",
        f"{threshold} a {threshold + width}",
        f"más de {threshold + width}",
    ]
    dashboard["rango_puntaje"] = pd.cut(
        dashboard[reference_column], bins=cuts, labels=labels, right=False
    )
    return dashboard

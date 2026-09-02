"""Ingeniería de variables (fase 03, secciones 3.9 y 3.10).

Dos pasos:

- `impute_profiling`: trata los nulos de las variables de **perfilado**. Las
  categóricas se imputan como `"Desconocido"` (categoría propia); `Quakers`
  con 0; las numéricas de perfilado (`altitude_mean_meters`, `Moisture`,
  `Harvest.Year`) **no se imputan** (no entran al clustering y no tiene
  sentido inventar una medición), pero se agrega un indicador de faltante.

- `build_clustering_matrix`: selecciona los atributos de catación que entran
  al agrupamiento (`config -> clustering.features`) y los estandariza con
  `StandardScaler`. El clustering agrupa por distancia, así que todas las
  columnas tienen que estar en la misma escala.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Grupos de variables del dataset reducido (ver notebook 03, sección 3.8).
FEATURE_GROUPS: dict[str, list[str]] = {
    "catacion": [
        "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
        "Uniformity", "Clean.Cup", "Sweetness", "Cupper.Points",
    ],
    "referencia": ["Total.Cup.Points"],
    "perfilado_origen": [
        "Country.of.Origin", "altitude_mean_meters", "Variety",
        "Processing.Method", "Harvest.Year",
    ],
    "perfilado_fisico": [
        "Color", "Moisture", "Category.One.Defects",
        "Category.Two.Defects", "Quakers",
    ],
}

# Categóricas de perfilado que se imputan como categoría "Desconocido".
CATEGORICAL_TO_FILL = ["Color", "Variety", "Processing.Method", "Country.of.Origin"]


def impute_profiling(df: pd.DataFrame, fill_value: str = "Desconocido") -> pd.DataFrame:
    """Imputa las categóricas de perfilado como `fill_value` y `Quakers` con 0;
    agrega `altitude_reportada` y `humedad_reportada` (1/0). Deja `NaN` en
    `altitude_mean_meters`, `Moisture` y `Harvest.Year` a propósito.
    """
    df = df.copy()
    for col in CATEGORICAL_TO_FILL:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna(fill_value)
    if "Quakers" in df.columns:
        df["Quakers"] = df["Quakers"].fillna(0)
    if "altitude_mean_meters" in df.columns:
        df["altitude_reportada"] = df["altitude_mean_meters"].notna().astype(int)
    if "Moisture" in df.columns:
        df["humedad_reportada"] = df["Moisture"].notna().astype(int)
    return df


def build_clustering_matrix(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    """Devuelve `(X, scaler)`, donde `X` es `df[features]` estandarizado
    (media 0, desviación 1 por columna) y `scaler` es el `StandardScaler`
    ajustado (para revertir la escala en la fase 06).
    """
    if df[features].isna().to_numpy().any():
        raise ValueError("Hay nulos en las columnas de clustering; no deberían existir.")
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features,
        index=df.index,
    )
    return X, scaler

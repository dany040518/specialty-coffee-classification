"""Limpieza de los datos (fase 03 - preparación).

Funciones simples, una por decisión tomada en
`notebooks/03_preparacion_de_los_datos.ipynb` (secciones 3.4 y 3.6). Los
parámetros (columnas a eliminar, rango de altitud, columnas cuyo 0 es
faltante) vienen de `config/config.yaml -> preparation`.

Cada función devuelve una copia transformada del DataFrame; ninguna modifica
el original.
"""

import numpy as np
import pandas as pd


def drop_corrupt_rows(df: pd.DataFrame, score_col: str = "Total.Cup.Points") -> pd.DataFrame:
    """Elimina filas cuyo puntaje total no es un puntaje de catación válido
    (`<= 0`). En la escala SCA los lotes evaluados no bajan de ~60; un 0 es
    un lote no evaluado o mal cargado (ver hallazgo en el notebook 02)."""
    return df.loc[df[score_col] > 0].reset_index(drop=True)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Elimina las columnas indicadas que existan en `df` (ignora las que no)."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def clip_altitude_to_nan(
    df: pd.DataFrame,
    valid_range: tuple[float, float],
    col: str = "altitude_mean_meters",
) -> pd.DataFrame:
    """Lleva a `NaN` los valores de altitud fuera de `(lo, hi]` metros
    (errores de digitación: máximos de decenas de miles de metros y valores
    de 1 m o menos)."""
    df = df.copy()
    lo, hi = valid_range
    df.loc[(df[col] <= lo) | (df[col] > hi), col] = np.nan
    return df


def zeros_to_nan(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convierte los `0` en `NaN` para las columnas indicadas (p. ej.
    `Moisture`, cuyos ceros son faltantes codificados como cero)."""
    df = df.copy()
    for col in columns:
        df.loc[df[col] == 0, col] = np.nan
    return df


def normalize_text_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el formato de las categóricas de texto: quita espacios
    sobrantes en `Variety` y `Processing.Method`. (No se hace `.title()`
    porque rompería nombres como `SL28`.)"""
    df = df.copy()
    for col in ("Variety", "Processing.Method"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def parse_harvest_year(df: pd.DataFrame, col: str = "Harvest.Year") -> pd.DataFrame:
    """Convierte `Harvest.Year` de texto libre (`2013/2014`, `March 2010`,
    `mmm`...) al primer año de 4 dígitos que aparezca; lo que no tiene año
    queda como `<NA>` (entero nullable)."""
    df = df.copy()
    anio = df[col].astype("string").str.extract(r"((?:19|20)\d{2})")[0]
    df[col] = pd.to_numeric(anio, errors="coerce").astype("Int64")
    return df


def clean_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aplica en orden toda la limpieza y reducción de columnas de la fase 03
    usando los parámetros de `config/config.yaml -> preparation`:

    1. elimina el registro corrupto (`Total.Cup.Points <= 0`),
    2. elimina identificadores y columnas de altitud redundantes (`drop_columns`),
    3. elimina `Species` (constante) y columnas administrativas (`drop_columns_admin`),
    4. recorta la altitud fuera de rango a `NaN`,
    5. convierte `Moisture == 0` en `NaN`,
    6. normaliza el texto de las categóricas,
    7. parsea `Harvest.Year` a año numérico.

    El resultado es el mismo dataset reducido que se construye paso a paso en
    el notebook 03 (secciones 3.4 a 3.8).
    """
    prep = config["preparation"]
    df = drop_corrupt_rows(df, config["validation"]["reference_column"])
    df = drop_columns(df, prep["drop_columns"])
    df = drop_columns(df, prep.get("drop_columns_admin", []))
    df = clip_altitude_to_nan(df, tuple(prep["altitude_valid_range"]))
    df = zeros_to_nan(df, prep.get("zero_as_missing", ["Moisture"]))
    df = normalize_text_categories(df)
    df = parse_harvest_year(df)
    return df

"""Utilidades genéricas reutilizadas por varios notebooks."""

import pandas as pd


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume la cantidad y el porcentaje de valores nulos por columna.

    Extraído del notebook original (Proyecto_Final_Machine_Learning.ipynb,
    celdas 4 y 8), donde esta misma lógica se repetía manualmente para
    df_arabica y df_robusta.

    Args:
        df: DataFrame a inspeccionar.

    Returns:
        DataFrame con columnas `cantidad_nulos` y `porcentaje`, ordenado
        de mayor a menor porcentaje de nulos.
    """
    valores_nulos = df.isnull().sum()
    porcentaje_nulos = (valores_nulos / len(df)) * 100
    return pd.DataFrame(
        {
            "cantidad_nulos": valores_nulos,
            "porcentaje": porcentaje_nulos.round(2),
        }
    ).sort_values("porcentaje", ascending=False)

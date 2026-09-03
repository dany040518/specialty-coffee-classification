"""Asignación de lotes nuevos a un grupo del clustering (fase 06).

Un lote nuevo no se puede pasar directo al k-means guardado, porque el modelo
final no se ajustó sobre los 7 atributos de catación sino sobre las 2 primeras
componentes principales. Hay que repetir con él los mismos tres pasos que se le
hicieron a los 1310 lotes de la base, y en el mismo orden:

1. estandarizar con la media y la escala de la fase 03
   (`data/processed/clustering_scaler.json`),
2. proyectar a `pca_2` con el PCA reconstruido en la fase 05
   (`models/clustering_pca.joblib`),
3. predecir el grupo con el k-means de la fase 04
   (`models/clustering_model.joblib`).

Saltarse el primero o el segundo devuelve un grupo, pero uno equivocado, porque
el lote quedaría en un espacio distinto al de los centros del modelo.

Una nota sobre qué esperar de esta función. Si se le pasan de vuelta los mismos
1310 lotes de la base, devuelve el grupo original en el 99.31 % de los casos, no
en el 100 %. Los 9 lotes que cambian son los que están justo en la frontera entre
dos grupos: su silueta va de -0.016 a 0.058 (o sea, prácticamente cero) y quedan
casi a la misma distancia de dos centros. La diferencia viene de que `labels_`
guarda la asignación de la última iteración del ajuste y `predict` recalcula
contra los centros finales. No es un error: es la misma inestabilidad de borde
que la fase 05 midió en unos 32 lotes, y confirma que las fronteras entre grupos
son convencionales.

El contenido supervisado anterior (`predict` / `predict_proba` de un pipeline de
clasificación) se eliminó al cambiar el proyecto a clustering.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import resolve_path


def assign_new_lotes(df_new: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Asigna lotes nuevos al grupo más cercano del modelo final.

    Args:
        df_new: DataFrame con los 7 atributos de catación de
            `config -> clustering.features`, en puntos SCA (0 a 10), sin
            estandarizar. Puede traer otras columnas, que se conservan.
        config: contenido de `config/config.yaml`.

    Returns:
        Una copia de `df_new` con la columna `grupo` agregada.

    Raises:
        ValueError: si falta alguno de los atributos de catación o si alguno
            viene con valores nulos, porque el modelo no puede asignar un grupo
            a un lote incompleto.
    """
    features = config["clustering"]["features"]
    processed_dir = resolve_path(config["paths"]["data"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])

    faltantes = [c for c in features if c not in df_new.columns]
    if faltantes:
        raise ValueError(f"faltan los atributos de catación: {faltantes}")
    if df_new[features].isna().any().any():
        raise ValueError("hay valores nulos en los atributos de catación")

    # 1) Estandarizar con los parámetros de la fase 03, no con los del lote nuevo:
    #    la media y la escala tienen que ser las mismas con las que se entrenó.
    escalador = json.loads(
        (processed_dir / "clustering_scaler.json").read_text(encoding="utf-8")
    )
    media = np.array(escalador["mean"])
    escala = np.array(escalador["scale"])
    X = (df_new[escalador["features"]].to_numpy() - media) / escala

    # 2) Proyectar al espacio donde vive el modelo (pca_2).
    pca = joblib.load(models_dir / "clustering_pca.joblib")
    X_pca = pca.transform(X)

    # 3) Predecir el grupo con el k-means ya ajustado.
    kmeans = joblib.load(models_dir / "clustering_model.joblib")
    resultado = df_new.copy()
    resultado["grupo"] = kmeans.predict(X_pca)
    return resultado

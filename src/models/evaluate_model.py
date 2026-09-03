"""Evaluación de los grupos del clustering (fase 05 - evaluación).

Funciones planas que reproducen lo que hace `notebooks/05_evaluacion.ipynb`:

- `internal_metrics`: las tres métricas de `config -> validation.internal_metrics`
  para una partición, en el espacio que se le pase (sección 5.4),
- `stability_by_seed`: reajusta k-means con varias semillas y mide con el índice
  de Rand ajustado (ARI) cuánto se parece cada corrida a la partición de
  referencia (sección 5.5),
- `stability_by_subsample`: repite el procedimiento completo (PCA + k-means)
  sobre submuestras aleatorias y mide el ARI y el porcentaje de lotes que
  conservan su grupo (secciones 5.6 y 5.7),
- `external_contrast`: cruza los grupos con `Total.Cup.Points >= 80`, columna que
  no entró al agrupamiento (sección 5.8).

El desglose de la silueta por grupo ya existe en `train_model.silhouette_by_group`
desde la fase 04, así que acá solo se reexporta para no escribirlo dos veces.

Los parámetros de las pruebas de estabilidad (número de semillas, número de
submuestras y fracción de cada una) salen de `config -> validation.stability`.

El contenido supervisado anterior (accuracy/precision/recall/F1/ROC-AUC, matriz
de confusión) se eliminó al cambiar el proyecto a clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.models.train_model import silhouette_by_group  # noqa: F401  (se reexporta)


def internal_metrics(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calcula las tres métricas internas de una partición sobre `matrix`.

    Las etiquetas son siempre las del modelo final; lo que cambia según el
    espacio que se pase es dónde se miden las distancias. El notebook llama a
    esta función dos veces, en `pca_2` y en los 7 atributos estandarizados,
    porque reducir dimensiones tiende a subir la silueta y los dos números no
    son directamente comparables (ver notebook 05, sección 5.2).

    Args:
        matrix: matriz de coordenadas de los lotes (n_lotes x n_dimensiones).
        labels: etiqueta de grupo de cada lote.

    Returns:
        dict con `silhouette`, `davies_bouldin` y `calinski_harabasz`.
    """
    return {
        "silhouette": round(silhouette_score(matrix, labels), 4),
        "davies_bouldin": round(davies_bouldin_score(matrix, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(matrix, labels), 1),
    }


def stability_by_seed(
    matrix: np.ndarray, labels_ref: np.ndarray, config: dict
) -> pd.DataFrame:
    """Reajusta k-means con distintas semillas y compara con la partición de referencia.

    K-means arranca con centros aleatorios, así que dos corridas pueden terminar
    en soluciones distintas. Se prueban las semillas 0 a `n_seeds - 1` sobre la
    misma matriz y se mide el ARI de cada una contra `labels_ref`. Un ARI de 1
    significa que la partición es idéntica, sin importar cómo quedaron numerados
    los grupos.

    Args:
        matrix: la misma matriz sobre la que se ajustó el modelo final.
        labels_ref: etiquetas del modelo final.
        config: contenido de `config/config.yaml`.

    Returns:
        DataFrame con una fila por corrida y las columnas `semilla` y `ari`.
    """
    k = config["clustering"]["final"]["k"]
    kmeans_params = config["clustering"]["algorithms"]["kmeans"]
    n_seeds = config["validation"]["stability"]["n_seeds"]

    rows = []
    for seed in range(n_seeds):
        labels = KMeans(n_clusters=k, random_state=seed, **kmeans_params).fit_predict(matrix)
        rows.append({"semilla": seed, "ari": round(adjusted_rand_score(labels_ref, labels), 4)})
    return pd.DataFrame(rows)


def stability_by_subsample(
    X: pd.DataFrame, labels_ref: np.ndarray, config: dict
) -> tuple[pd.DataFrame, np.ndarray]:
    """Repite el agrupamiento sobre submuestras y mide cuánto se mantiene.

    En cada una de las `n_subsamples` submuestras se toma una fracción
    `subsample_fraction` de los lotes y se repite el procedimiento completo, PCA
    incluido: dejar el PCA fijo esconderría la mitad de la variabilidad que se
    quiere medir. El ARI se calcula contra las etiquetas del modelo final
    restringidas a los lotes de esa submuestra.

    Para contar cuántos lotes conservan su grupo hay que emparejar primero las
    dos numeraciones, porque los algoritmos numeran los grupos de forma
    arbitraria: cada grupo nuevo se empareja con el grupo original con el que
    más lotes comparte (`pd.crosstab` + `idxmax`).

    Args:
        X: matriz de entrada del clustering, sin reducir (fase 03).
        labels_ref: etiquetas del modelo final, para los mismos lotes de `X`.
        config: contenido de `config/config.yaml`.

    Returns:
        Tupla ``(resumen, pct_por_lote)``: el resumen es un DataFrame con una
        fila por submuestra y las columnas `iteracion`, `ari` y `pct_conserva`
        (porcentaje de lotes de esa submuestra que conservaron su grupo), y
        `pct_por_lote` es el porcentaje de veces que cada lote conservó su grupo,
        en el mismo orden que `labels_ref`.
    """
    seed = config["project"]["random_seed"]
    k = config["clustering"]["final"]["k"]
    kmeans_params = config["clustering"]["algorithms"]["kmeans"]
    stability = config["validation"]["stability"]

    n_rows = len(X)
    size = int(stability["subsample_fraction"] * n_rows)
    generator = np.random.default_rng(seed)

    times_included = np.zeros(n_rows)
    times_kept = np.zeros(n_rows)
    rows = []

    for iteration in range(stability["n_subsamples"]):
        indices = generator.choice(n_rows, size=size, replace=False)
        matrix = PCA(n_components=2, random_state=seed).fit_transform(X.iloc[indices])
        labels = KMeans(n_clusters=k, random_state=seed, **kmeans_params).fit_predict(matrix)

        labels_original = labels_ref[indices]
        # Emparejamiento de las dos numeraciones antes de comparar lote a lote.
        equivalence = pd.crosstab(labels, labels_original).idxmax(axis=1)
        labels_translated = pd.Series(labels).map(equivalence).to_numpy()
        matches = labels_translated == labels_original

        rows.append(
            {
                "iteracion": iteration,
                "ari": round(adjusted_rand_score(labels_original, labels), 4),
                "pct_conserva": matches.mean() * 100,
            }
        )
        times_included[indices] += 1
        times_kept[indices] += matches

    return pd.DataFrame(rows), times_kept / times_included * 100


def external_contrast(
    df: pd.DataFrame, labels: np.ndarray, config: dict
) -> tuple[pd.DataFrame, float]:
    """Cruza los grupos con la convención de la industria (café de especialidad).

    `validation.reference_column` (`Total.Cup.Points`) no entró al agrupamiento,
    así que sirve de referencia externa: si los grupos se ordenan según ese
    puntaje sin haberlo visto, capturaron calidad y no ruido. El ARI dice, además,
    si el clustering reprodujo el corte de `validation.specialty_threshold`
    puntos o encontró algo distinto.

    Args:
        df: dataset con la columna de referencia (`coffee_clustered.csv`).
        labels: etiqueta de grupo de cada lote.
        config: contenido de `config/config.yaml`.

    Returns:
        Tupla ``(tabla, ari)``: la tabla cruzada grupo x especialidad con los
        conteos, el porcentaje por fila y el `n` de cada grupo, y el ARI entre la
        partición de grupos y la binaria del umbral.
    """
    reference_column = config["validation"]["reference_column"]
    threshold = config["validation"]["specialty_threshold"]

    specialty = df[reference_column] >= threshold
    counts = pd.crosstab(pd.Series(labels, index=df.index, name="grupo"), specialty)
    counts.columns = [f"< {threshold}", f">= {threshold}"]
    percentages = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)

    table = pd.concat([counts, percentages.add_suffix(" (%)")], axis=1)
    table["n"] = counts.sum(axis=1)
    return table, adjusted_rand_score(labels, specialty.astype(int))

"""Modelado del clustering (fase 04 - modelado).

Funciones planas que reproducen lo que hace `notebooks/04_modelado.ipynb`:

- `build_representations`: arma las tres representaciones de la matriz de
  entrada que se comparan (los 7 atributos estandarizados y dos reducciones
  por PCA),
- `fit_predict`: ajusta uno de los algoritmos de
  `config -> clustering.algorithms` y devuelve la etiqueta de grupo de cada lote,
- `sweep`: recorre representación x algoritmo x k y calcula las métricas
  internas de cada combinación,
- `fit_final_model`: ajusta el modelo elegido en `config -> clustering.final`
  y lo devuelve junto con las etiquetas,
- `silhouette_by_group`: desglosa la silueta por grupo, para diagnóstico.

No hay clases base ni registro de modelos: el proyecto compara tres algoritmos
concretos de scikit-learn y no necesita más abstracción. El contenido
supervisado anterior (regresión logística + SMOTE) se eliminó al cambiar el
proyecto a clustering no supervisado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def build_representations(X: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    """Devuelve las representaciones de la matriz de entrada que se comparan.

    - ``"7_atributos"``: los atributos ya estandarizados, sin reducir.
    - ``"pca_2"`` / ``"pca_4"``: las 2 y 4 primeras componentes principales.

    Los nombres coinciden con `config -> clustering.pca.representations` y con
    los del notebook 04 (sección 4.5).
    """
    seed = config["project"]["random_seed"]
    matrix = X.to_numpy()
    return {
        "7_atributos": matrix,
        "pca_2": PCA(n_components=2, random_state=seed).fit_transform(matrix),
        "pca_4": PCA(n_components=4, random_state=seed).fit_transform(matrix),
    }


def _build_estimator(algorithm: str, k: int, config: dict):
    """Instancia (sin ajustar) el algoritmo pedido con los parámetros del config."""
    seed = config["project"]["random_seed"]
    params = config["clustering"]["algorithms"].get(algorithm) or {}
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, **params)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, **params)
    if algorithm == "gaussian_mixture":
        return GaussianMixture(n_components=k, random_state=seed, **params)
    raise ValueError(f"algoritmo no soportado: {algorithm!r}")


def fit_predict(algorithm: str, k: int, matrix: np.ndarray, config: dict) -> np.ndarray:
    """Ajusta `algorithm` con `k` grupos sobre `matrix` y devuelve las etiquetas."""
    return _build_estimator(algorithm, k, config).fit_predict(matrix)


def sweep(representations: dict[str, np.ndarray], config: dict) -> pd.DataFrame:
    """Recorre representación x algoritmo x k y calcula sus métricas internas.

    Devuelve un DataFrame con una fila por combinación y las columnas
    `representacion`, `algoritmo`, `k`, `silhouette`, `davies_bouldin`,
    `calinski_harabasz` (mismo contenido que `reports/tables/clustering_comparison.csv`).
    """
    algorithms = list(config["clustering"]["algorithms"])
    k_min, k_max = config["clustering"]["k_min"], config["clustering"]["k_max"]
    rows = []
    for rep_name, matrix in representations.items():
        for algorithm in algorithms:
            for k in range(k_min, k_max + 1):
                labels = fit_predict(algorithm, k, matrix, config)
                rows.append(
                    {
                        "representacion": rep_name,
                        "algoritmo": algorithm,
                        "k": k,
                        "silhouette": round(silhouette_score(matrix, labels), 3),
                        "davies_bouldin": round(davies_bouldin_score(matrix, labels), 3),
                        "calinski_harabasz": round(calinski_harabasz_score(matrix, labels), 0),
                    }
                )
    return pd.DataFrame(rows)


def fit_final_model(representations: dict[str, np.ndarray], config: dict):
    """Ajusta el modelo elegido en `config -> clustering.final`.

    Returns:
        Tupla ``(estimator ajustado, matriz usada, etiquetas de grupo)``.
    """
    final = config["clustering"]["final"]
    matrix = representations[final["representation"]]
    estimator = _build_estimator(final["algorithm"], final["k"], config)
    estimator.fit(matrix)
    labels = (
        estimator.labels_
        if hasattr(estimator, "labels_")
        else estimator.predict(matrix)
    )
    return estimator, matrix, labels


def silhouette_by_group(matrix: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Desglosa la silueta por grupo: media, mínimo, p25, % negativa y n."""
    sil = silhouette_samples(matrix, labels)
    return (
        pd.DataFrame({"grupo": labels, "silueta": sil})
        .groupby("grupo")["silueta"]
        .agg(
            media="mean",
            minimo="min",
            p25=lambda s: s.quantile(0.25),
            pct_negativa=lambda s: round((s < 0).mean() * 100, 1),
            n="count",
        )
        .round(3)
    )

"""Fachada del pipeline CRISP-DM.

Es el único módulo que los notebooks necesitan importar para ejecutar una
fase completa de forma reproducible. Hoy cubre las **fases 03** (preparación
de los datos), **04** (modelado del clustering) y **05** (evaluación de los
grupos).

- `prepare_data(config)` reproduce todo lo que el notebook
  `03_preparacion_de_los_datos.ipynb` hace paso a paso: carga → limpieza y
  reducción de columnas → imputación de perfilado → construcción y
  estandarización de la matriz de clustering → guardado en `data/processed/`.
- `train_clustering(config)` reproduce el notebook `04_modelado.ipynb`:
  representaciones (7 atributos / PCA) → barrido representación x algoritmo x k
  → ajuste del modelo elegido en `config -> clustering.final` → guardado del
  modelo, las etiquetas y la tabla comparativa.
- `evaluate_clustering(config)` reproduce el notebook `05_evaluacion.ipynb`:
  verificación de reproducibilidad → validación interna en los dos espacios →
  estabilidad ante semillas y submuestras → contraste externo contra
  `Total.Cup.Points >= 80` → guardado de las tablas y del resumen numérico.
  Las figuras de diagnóstico quedan solo en el notebook, igual que en la fase 04.
"""

import json
from typing import Any

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

from src.config import resolve_path
from src.data.clean_data import clean_dataset
from src.data.load_data import load_raw_data
from src.features.build_features import build_clustering_matrix, impute_profiling
from src.models.evaluate_model import (
    external_contrast,
    internal_metrics,
    silhouette_by_group,
    stability_by_seed,
    stability_by_subsample,
)
from src.models.train_model import build_representations, fit_final_model, sweep


def load_dataset(config: dict) -> pd.DataFrame:
    """Carga la especie definida en `config -> dataset.species`.

    En la fase 03 se decidió trabajar solo con arabica (ver notebook 03,
    sección 3.5). Si en el futuro se soporta "combined", el mapeo de los
    formularios sensoriales se resolvería aquí.
    """
    species = config["dataset"]["species"]
    if species == "arabica":
        return load_raw_data("arabica", config)
    raise NotImplementedError(
        f"dataset.species = {species!r}. Solo se soporta 'arabica' (ver notebook 03, 3.5)."
    )


def prepare_data(config: dict, save: bool = True) -> dict[str, Any]:
    """Ejecuta toda la preparación de la fase 03.

    Args:
        config: contenido de `config/config.yaml`.
        save: si True, guarda los artefactos en `data/processed/`.

    Returns:
        dict con `df` (dataset reducido e imputado), `X` (matriz de clustering
        estandarizada), `scaler` (StandardScaler ajustado) y `n_rows`.
    """
    raw = load_dataset(config)
    df = clean_dataset(raw, config)
    df = impute_profiling(df)

    features = config["clustering"]["features"]
    X, scaler = build_clustering_matrix(df, features)

    if save:
        processed_dir = resolve_path(config["paths"]["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        X.to_csv(processed_dir / "clustering_input.csv", index=False)
        df.to_csv(processed_dir / "coffee_clean.csv", index=False)
        (processed_dir / "clustering_scaler.json").write_text(
            json.dumps(
                {
                    "features": list(features),
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return {"df": df, "X": X, "scaler": scaler, "n_rows": len(df)}


def train_clustering(config: dict, save: bool = True) -> dict[str, Any]:
    """Ejecuta todo el modelado de la fase 04.

    Parte de los artefactos de la fase 03 (`data/processed/clustering_input.csv`
    y `coffee_clean.csv`), compara las combinaciones de representación x
    algoritmo x k, ajusta el modelo elegido en `config -> clustering.final` y
    guarda el modelo, la etiqueta de grupo de cada lote y la tabla comparativa.

    Args:
        config: contenido de `config/config.yaml`.
        save: si True, guarda los artefactos en `models/`, `data/processed/` y
            `reports/tables/`.

    Returns:
        dict con `model` (estimador ajustado), `labels` (array de grupos),
        `comparison` (DataFrame del barrido) y `metadata` (dict del modelo elegido).
    """
    processed_dir = resolve_path(config["paths"]["data"]["processed_dir"])
    X = pd.read_csv(processed_dir / "clustering_input.csv")
    coffee = pd.read_csv(processed_dir / "coffee_clean.csv")

    representations = build_representations(X, config)
    comparison = sweep(representations, config)
    model, matrix, labels = fit_final_model(representations, config)

    final = config["clustering"]["final"]
    metadata = {
        "representacion": final["representation"],
        "algoritmo": final["algorithm"],
        "k": int(final["k"]),
        "random_seed": config["project"]["random_seed"],
        "n_lotes": int(len(labels)),
        "silhouette": float(round(silhouette_score(matrix, labels), 4)),
        "davies_bouldin": float(round(davies_bouldin_score(matrix, labels), 4)),
        "calinski_harabasz": float(round(calinski_harabasz_score(matrix, labels), 1)),
        "tam_grupos": pd.Series(labels).value_counts().sort_index().to_dict(),
    }

    if save:
        import joblib

        models_dir = resolve_path(config["paths"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = resolve_path(config["paths"]["reports"]["tables_dir"])
        tables_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, models_dir / "clustering_model.joblib")
        clustered = coffee.copy()
        clustered["grupo"] = labels
        clustered.to_csv(processed_dir / "coffee_clustered.csv", index=False)
        comparison.to_csv(tables_dir / "clustering_comparison.csv", index=False)
        (models_dir / "clustering_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    return {
        "model": model,
        "labels": labels,
        "comparison": comparison,
        "metadata": metadata,
    }


def evaluate_clustering(config: dict, save: bool = True) -> dict[str, Any]:
    """Ejecuta toda la evaluación de la fase 05.

    Parte de los artefactos de las fases 03 y 04
    (`data/processed/clustering_input.csv`, `data/processed/coffee_clustered.csv`
    y `models/clustering_model.joblib`), reconstruye el PCA que la fase 04 no
    guardó, comprueba que el modelo se reproduce, recalcula las métricas
    internas en los dos espacios, mide la estabilidad ante semillas y
    submuestras y contrasta los grupos contra `Total.Cup.Points >= 80`.

    Args:
        config: contenido de `config/config.yaml`.
        save: si True, guarda los artefactos en `models/` y `reports/tables/`.

    Returns:
        dict con `labels` (etiquetas reproducidas), `internal` (DataFrame de
        métricas en los dos espacios), `silhouette_groups` (silueta por grupo),
        `stability_seed`, `stability_subsample`, `contrast` (tabla cruzada) y
        `evaluation` (dict del resumen numérico de la fase).
    """
    seed = config["project"]["random_seed"]
    final = config["clustering"]["final"]
    kmeans_params = config["clustering"]["algorithms"]["kmeans"]
    reference_column = config["validation"]["reference_column"]
    threshold = config["validation"]["specialty_threshold"]
    stability = config["validation"]["stability"]

    processed_dir = resolve_path(config["paths"]["data"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    X = pd.read_csv(processed_dir / "clustering_input.csv")
    coffee = pd.read_csv(processed_dir / "coffee_clustered.csv")

    # El modelo final se ajustó sobre pca_2, pero la fase 04 solo guardó el
    # k-means: sin el PCA no se puede proyectar un lote nuevo (ver notebook 05, 5.3).
    pca = PCA(n_components=2, random_state=seed)
    matrix_pca2 = pca.fit_transform(X)
    labels = KMeans(n_clusters=final["k"], random_state=seed, **kmeans_params).fit_predict(
        matrix_pca2
    )
    labels_saved = coffee["grupo"].to_numpy()
    reproducibility_ari = round(adjusted_rand_score(labels_saved, labels), 4)

    # Validación interna en los dos espacios (5.4).
    spaces = {
        "pca_2 (donde se ajustó)": matrix_pca2,
        "7 atributos estandarizados": X.to_numpy(),
    }
    rows = []
    for name, matrix in spaces.items():
        row = {"espacio": name}
        row.update(internal_metrics(matrix, labels))
        rows.append(row)
    internal = pd.DataFrame(rows)
    silhouette_groups = silhouette_by_group(matrix_pca2, labels)

    # Estabilidad (5.5 a 5.7).
    stability_seed = stability_by_seed(matrix_pca2, labels, config)
    stability_subsample, pct_by_row = stability_by_subsample(X, labels, config)
    stability_groups = (
        pd.DataFrame({"grupo": labels, "pct_conserva": pct_by_row})
        .groupby("grupo")["pct_conserva"]
        .agg(media="mean", minimo="min", p25=lambda s: s.quantile(0.25), n="count")
        .round(2)
    )

    # Contraste externo (5.8) y la comparación de k = 2 (5.9).
    contrast, ari_groups_threshold = external_contrast(coffee, labels, config)
    specialty = coffee[reference_column] >= threshold
    score_summary = (
        coffee.groupby("grupo")[reference_column]
        .agg(n="count", media="mean", desviacion="std", minimo="min", maximo="max")
        .round(2)
    )
    score_summary["pct_especialidad"] = (
        specialty.groupby(coffee["grupo"]).mean() * 100
    ).round(1)
    labels_k2 = KMeans(
        n_clusters=config["clustering"]["k_min"], random_state=seed, **kmeans_params
    ).fit_predict(matrix_pca2)

    evaluation = {
        "representacion": final["representation"],
        "algoritmo": final["algorithm"],
        "k": int(final["k"]),
        "random_seed": seed,
        "n_lotes": int(len(X)),
        "reproducibilidad_ari": float(reproducibility_ari),
        "metricas_pca_2": internal.iloc[0].drop("espacio").astype(float).round(4).to_dict(),
        "metricas_7_atributos": internal.iloc[1].drop("espacio").astype(float).round(4).to_dict(),
        "silueta_por_grupo": silhouette_groups["media"].astype(float).to_dict(),
        "estabilidad_semillas": {
            "n_seeds": int(stability["n_seeds"]),
            "ari_medio": float(round(stability_seed["ari"].mean(), 4)),
            "ari_minimo": float(round(stability_seed["ari"].min(), 4)),
            "ari_maximo": float(round(stability_seed["ari"].max(), 4)),
        },
        "estabilidad_submuestras": {
            "n_subsamples": int(stability["n_subsamples"]),
            "subsample_fraction": float(stability["subsample_fraction"]),
            "ari_medio": float(round(stability_subsample["ari"].mean(), 4)),
            "ari_desviacion": float(round(stability_subsample["ari"].std(), 4)),
            "ari_minimo": float(round(stability_subsample["ari"].min(), 4)),
            "ari_maximo": float(round(stability_subsample["ari"].max(), 4)),
        },
        "estabilidad_asignacion": {
            "pct_conserva_medio": float(round(stability_subsample["pct_conserva"].mean(), 2)),
            "pct_conserva_por_grupo": stability_groups["media"].astype(float).to_dict(),
            "lotes_inestables_bajo_80pct": int((pct_by_row < 80).sum()),
        },
        "contraste_externo": {
            "columna": reference_column,
            "umbral": int(threshold),
            "pct_global_especialidad": float(round(specialty.mean() * 100, 1)),
            "ari_k3_vs_umbral": float(round(ari_groups_threshold, 4)),
            "ari_k2_vs_umbral": float(
                round(adjusted_rand_score(labels_k2, specialty.astype(int)), 4)
            ),
            "puntaje_medio_por_grupo": score_summary["media"].astype(float).to_dict(),
            "pct_especialidad_por_grupo": score_summary["pct_especialidad"].astype(float).to_dict(),
        },
    }

    if save:
        import joblib

        models_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = resolve_path(config["paths"]["reports"]["tables_dir"])
        tables_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(pca, models_dir / "clustering_pca.joblib")

        # 1) validación interna: métricas en los dos espacios + silueta por grupo.
        rows = []
        for _, row in internal.iterrows():
            for metric in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
                rows.append(
                    {
                        "bloque": "metricas_globales",
                        "item": row["espacio"],
                        "metrica": metric,
                        "valor": row[metric],
                    }
                )
        for group, row in silhouette_groups.iterrows():
            for metric in silhouette_groups.columns:
                rows.append(
                    {
                        "bloque": "silueta_por_grupo",
                        "item": f"grupo {group}",
                        "metrica": metric,
                        "valor": row[metric],
                    }
                )
        pd.DataFrame(rows).to_csv(tables_dir / "evaluacion_interna.csv", index=False)

        # 2) estabilidad: resumen de las tres pruebas.
        rows = [
            {"prueba": "semillas", "metrica": "n_corridas", "valor": stability["n_seeds"]},
            {"prueba": "semillas", "metrica": "ari_medio", "valor": round(stability_seed["ari"].mean(), 4)},
            {"prueba": "semillas", "metrica": "ari_minimo", "valor": round(stability_seed["ari"].min(), 4)},
            {"prueba": "semillas", "metrica": "ari_maximo", "valor": round(stability_seed["ari"].max(), 4)},
            {"prueba": "submuestras", "metrica": "n_corridas", "valor": stability["n_subsamples"]},
            {"prueba": "submuestras", "metrica": "fraccion", "valor": stability["subsample_fraction"]},
            {"prueba": "submuestras", "metrica": "ari_medio", "valor": round(stability_subsample["ari"].mean(), 4)},
            {"prueba": "submuestras", "metrica": "ari_desviacion", "valor": round(stability_subsample["ari"].std(), 4)},
            {"prueba": "submuestras", "metrica": "ari_minimo", "valor": round(stability_subsample["ari"].min(), 4)},
            {"prueba": "submuestras", "metrica": "ari_maximo", "valor": round(stability_subsample["ari"].max(), 4)},
            {"prueba": "asignacion", "metrica": "pct_conserva_medio", "valor": round(float(np.mean(stability_subsample["pct_conserva"])), 2)},
            {"prueba": "asignacion", "metrica": "pct_conserva_minimo", "valor": round(float(np.min(stability_subsample["pct_conserva"])), 2)},
            {"prueba": "asignacion", "metrica": "pct_conserva_maximo", "valor": round(float(np.max(stability_subsample["pct_conserva"])), 2)},
            {"prueba": "asignacion", "metrica": "lotes_inestables_bajo_80pct", "valor": int((pct_by_row < 80).sum())},
        ]
        for group, row in stability_groups.iterrows():
            rows.append(
                {
                    "prueba": "asignacion_por_grupo",
                    "metrica": f"pct_conserva_grupo_{group}",
                    "valor": row["media"],
                }
            )
        pd.DataFrame(rows).to_csv(tables_dir / "estabilidad.csv", index=False)

        # 3) contraste externo: la tabla cruzada grupo x especialidad.
        contrast.reset_index().to_csv(tables_dir / "contraste_externo.csv", index=False)

        (models_dir / "clustering_evaluation.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "labels": labels,
        "internal": internal,
        "silhouette_groups": silhouette_groups,
        "stability_seed": stability_seed,
        "stability_subsample": stability_subsample,
        "contrast": contrast,
        "evaluation": evaluation,
    }

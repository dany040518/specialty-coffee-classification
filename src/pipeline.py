"""Fachada del pipeline CRISP-DM.

Es el único módulo que los notebooks necesitan importar para ejecutar una
fase completa de forma reproducible. Hoy cubre las **fases 03** (preparación
de los datos) y **04** (modelado del clustering); la fase 05 se agregará aquí
cuando se implemente.

- `prepare_data(config)` reproduce todo lo que el notebook
  `03_preparacion_de_los_datos.ipynb` hace paso a paso: carga → limpieza y
  reducción de columnas → imputación de perfilado → construcción y
  estandarización de la matriz de clustering → guardado en `data/processed/`.
- `train_clustering(config)` reproduce el notebook `04_modelado.ipynb`:
  representaciones (7 atributos / PCA) → barrido representación x algoritmo x k
  → ajuste del modelo elegido en `config -> clustering.final` → guardado del
  modelo, las etiquetas y la tabla comparativa.
"""

import json
from typing import Any

import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import resolve_path
from src.data.clean_data import clean_dataset
from src.data.load_data import load_raw_data
from src.features.build_features import build_clustering_matrix, impute_profiling
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

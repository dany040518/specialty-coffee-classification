"""Fachada del pipeline CRISP-DM.

Es el único módulo que los notebooks necesitan importar para ejecutar una
fase completa de forma reproducible. Hoy cubre la **fase 03** (preparación de
los datos); las fases 04 y 05 se agregarán aquí cuando se implementen.

`prepare_data(config)` reproduce, en una sola llamada, todo lo que el
notebook `03_preparacion_de_los_datos.ipynb` hace paso a paso: carga →
limpieza y reducción de columnas → imputación de perfilado → construcción y
estandarización de la matriz de clustering → guardado de artefactos en
`data/processed/`.
"""

import json
from typing import Any

import pandas as pd

from src.config import resolve_path
from src.data.clean_data import clean_dataset
from src.data.load_data import load_raw_data
from src.features.build_features import build_clustering_matrix, impute_profiling


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

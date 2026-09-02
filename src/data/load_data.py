"""Carga de los datasets crudos de la Coffee Quality Institute (CQI).

Función única `load_raw_data(species, config)`: lee el CSV de una especie
tal como se distribuye, sin ninguna limpieza.
"""

import pandas as pd

from src.config import load_config, resolve_path

VALID_SPECIES = ("arabica", "robusta")


def load_raw_data(species: str, config: dict | None = None) -> pd.DataFrame:
    """Lee `data/raw/<species>_data_cleaned.csv` (arabica o robusta).

    Args:
        species: "arabica" o "robusta".
        config: config.yaml ya cargado; si es None se carga con `load_config()`.

    Returns:
        DataFrame crudo, sin transformaciones.
    """
    if species not in VALID_SPECIES:
        raise ValueError(f"species debe ser 'arabica' o 'robusta', recibido: {species!r}")
    config = config if config is not None else load_config()
    csv_path = resolve_path(config["paths"]["data"][f"{species}_raw"])
    return pd.read_csv(csv_path)

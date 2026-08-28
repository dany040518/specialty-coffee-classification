"""Carga de los datasets crudos (Coffee Quality Institute).

Extraído del notebook original (celdas 1 y 5), donde `pd.read_csv` se
llamaba de forma independiente para arabica y robusta con rutas fijas.
"""

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path


def load_raw_data(species: str, config: dict | None = None) -> pd.DataFrame:
    """Carga el dataset crudo de una especie de café ('arabica' o 'robusta').

    Args:
        species: "arabica" o "robusta".
        config: configuración ya cargada (dict de config.yaml). Si es None,
            se carga automáticamente con `load_config()`.

    Returns:
        DataFrame con los datos crudos, tal como se distribuyen en
        data/raw/ (sin ninguna limpieza aplicada).

    Raises:
        ValueError: si `species` no es "arabica" ni "robusta".
    """
    if config is None:
        config = load_config()

    key = f"{species}_raw"
    if key not in config["paths"]["data"]:
        raise ValueError(f"species debe ser 'arabica' o 'robusta', recibido: {species!r}")

    csv_path: Path = resolve_path(config["paths"]["data"][key])
    return pd.read_csv(csv_path)

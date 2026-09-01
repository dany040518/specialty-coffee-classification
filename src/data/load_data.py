"""Carga de los datasets crudos (Coffee Quality Institute).

Extraído del notebook original (celdas 1 y 5), donde `pd.read_csv` se
llamaba de forma independiente para arabica y robusta con rutas fijas.

`CQIRawDataLoader` implementa la interfaz `DataLoader` (`src/data/base.py`):
una instancia = una especie. Así, `src/pipeline.py` puede combinar varios
loaders sin conocer los detalles de CQI (ISP: la interfaz solo expone
`load()`).
"""

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data.base import DataLoader

VALID_SPECIES = ("arabica", "robusta")


class CQIRawDataLoader(DataLoader):
    """Carga el CSV crudo de una especie de café ('arabica' o 'robusta')."""

    def __init__(self, species: str, config: dict | None = None):
        if species not in VALID_SPECIES:
            raise ValueError(f"species debe ser 'arabica' o 'robusta', recibido: {species!r}")
        self.species = species
        self.config = config if config is not None else load_config()

    @property
    def csv_path(self) -> Path:
        key = f"{self.species}_raw"
        return resolve_path(self.config["paths"]["data"][key])

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)


def load_raw_data(species: str, config: dict | None = None) -> pd.DataFrame:
    """Wrapper funcional sobre `CQIRawDataLoader`, por compatibilidad con el
    notebook 02 (que ya llama a esta función) y para uso rápido fuera de la
    capa de orquestación.

    Args:
        species: "arabica" o "robusta".
        config: configuración ya cargada (dict de config.yaml). Si es None,
            se carga automáticamente con `load_config()`.

    Returns:
        DataFrame con los datos crudos, sin ninguna limpieza aplicada.
    """
    return CQIRawDataLoader(species, config).load()

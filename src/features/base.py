"""Interfaz para la capa de ingeniería de variables."""

from abc import ABC, abstractmethod

import pandas as pd


class FeatureBuilder(ABC):
    """Agrega o transforma columnas de un DataFrame con una única
    responsabilidad (por ejemplo, construir la variable objetivo).

    Nuevas variables derivadas se agregan como nuevas implementaciones de
    esta interfaz, sin modificar las existentes (OCP).
    """

    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

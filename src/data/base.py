"""Interfaces (capa de abstracción) para la capa de datos.

Definir estas interfaces permite que `src/pipeline.py` (la capa de
orquestación) dependa de abstracciones y no de implementaciones concretas
(Dependency Inversion Principle), y que se puedan agregar nuevas fuentes de
datos o nuevos pasos de limpieza sin modificar el código existente
(Open/Closed Principle).
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataLoader(ABC):
    """Fuente de datos crudos. Cada implementación sabe cargar UN origen."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Devuelve el DataFrame crudo, tal como se distribuye la fuente."""
        raise NotImplementedError


class DataCleaningStep(ABC):
    """Un paso de limpieza atómico, con una única responsabilidad (SRP).

    `CleaningPipeline` (en `clean_data.py`) compone una lista ordenada de
    pasos que implementan esta interfaz (patrón Composite), de modo que
    agregar un nuevo tratamiento de datos no requiere tocar los pasos ya
    existentes ni la clase que los orquesta (OCP).
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recibe un DataFrame y devuelve una copia transformada."""
        raise NotImplementedError

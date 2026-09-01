"""Limpieza de los datos (fase 03 - preparación de los datos).

Cada paso de limpieza implementa `DataCleaningStep` (`src/data/base.py`) y
tiene una única responsabilidad (SRP). `CleaningPipeline` los compone en
orden (patrón Composite): agregar un paso nuevo (por ejemplo, un tratamiento
de outliers) no requiere modificar los pasos existentes (OCP).

Decisiones tomadas para esta iteración (documentadas también en
`notebooks/03_preparacion_de_los_datos.ipynb` y `config/config.yaml`):

- Se elimina el registro con `Total.Cup.Points <= 0` detectado en la fase 02
  (dato corrupto / lote no evaluado, no una observación válida de baja
  calidad — ver hallazgo en `02_comprension_de_los_datos.ipynb`).
- No se eliminan filas duplicadas porque la fase 02 confirmó que no existen
  en ninguno de los dos datasets; el paso se deja implementado por
  completitud (para datasets futuros) pero no debería tener efecto hoy.
"""

import pandas as pd

from src.data.base import DataCleaningStep


class DropDuplicateRows(DataCleaningStep):
    """Elimina filas completamente duplicadas."""

    def __init__(self):
        self.n_dropped_: int | None = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        result = df.drop_duplicates().reset_index(drop=True)
        self.n_dropped_ = n_before - len(result)
        return result


class DropCorruptScores(DataCleaningStep):
    """Elimina filas cuyo puntaje total no es un puntaje de catación válido.

    La escala SCA va de 0 a 100 como suma de subpuntajes; en la práctica los
    lotes evaluados rara vez bajan de 60, así que un `Total.Cup.Points <= 0`
    se trata como dato corrupto (ver hallazgo en notebook 02), no como una
    observación real de baja calidad.
    """

    def __init__(self, score_column: str = "Total.Cup.Points", min_valid_score: float = 0.0):
        self.score_column = score_column
        self.min_valid_score = min_valid_score
        self.n_dropped_: int | None = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask_valid = df[self.score_column] > self.min_valid_score
        self.n_dropped_ = int((~mask_valid).sum())
        return df.loc[mask_valid].reset_index(drop=True)


class CleaningPipeline:
    """Aplica una lista ordenada de `DataCleaningStep` sobre un DataFrame."""

    def __init__(self, steps: list[DataCleaningStep]):
        self.steps = steps

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.apply(df)
        return df

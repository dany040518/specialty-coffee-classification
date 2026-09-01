"""Ingeniería de variables y construcción de la variable objetivo (fase 03).

Decisiones tomadas para esta iteración (ver `config/config.yaml -> target`
y `-> features`, y la discusión en `notebooks/03_preparacion_de_los_datos.ipynb`):

- **Variable objetivo:** `especialidad = 1` si `Total.Cup.Points >= 80`
  (umbral estándar SCA), `0` en caso contrario. Se calcula después de que
  `src/data/clean_data.py` ya eliminó el registro con puntaje corrupto.
- **Variables predictoras:** se usan únicamente variables de *contexto*
  (origen, variedad, método de procesamiento, altitud, color, especie,
  tamaño del lote, defectos, humedad) y explícitamente **no** los
  subpuntajes de catación (`Aroma`, `Flavor`, `Acidity`, ...), porque esos
  subpuntajes son los componentes que suman `Total.Cup.Points`: usarlos como
  predictores del umbral derivado de esa misma suma sería fuga de
  información (el modelo aprendería casi literalmente "sumar y comparar"
  en lugar de asociar condiciones de origen/cultivo con la clasificación,
  que es el objetivo de negocio declarado en `01_comprension_del_negocio.ipynb`).
- **Variedad/Método/Color/Altitud:** con nulos (hasta 89% en `Variety` de
  robusta) se imputan como categoría `"Unknown"` (categóricas) o mediana
  (`altitude_mean_meters`), en vez de descartar filas o columnas completas.
"""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.base import FeatureBuilder

# Variables de contexto usadas como predictoras (ver justificación arriba).
# Deliberadamente NO incluyen los subpuntajes de catación ni Total.Cup.Points.
NUMERIC_FEATURES = [
    "Number.of.Bags",
    "altitude_mean_meters",
    "Moisture",
    "Category.One.Defects",
    "Category.Two.Defects",
    "Quakers",
]
CATEGORICAL_FEATURES = [
    "Country.of.Origin",
    "Variety",
    "Processing.Method",
    "Color",
    "species",
]


class SpecialtyTargetBuilder(FeatureBuilder):
    """Construye la variable binaria 'café de especialidad' a partir de
    `Total.Cup.Points` y el umbral definido en `config.yaml -> target`.
    """

    def __init__(self, source_column: str, target_column: str, threshold: float):
        self.source_column = source_column
        self.target_column = target_column
        self.threshold = threshold

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.target_column] = (df[self.source_column] >= self.threshold).astype(int)
        return df

    @classmethod
    def from_config(cls, config: dict) -> "SpecialtyTargetBuilder":
        target_cfg = config["target"]
        return cls(
            source_column=target_cfg["source_column"],
            target_column=target_cfg["target_column"],
            threshold=target_cfg["specialty_threshold"],
        )


@dataclass
class FeatureSchema:
    """Agrupa qué columnas son numéricas/categóricas, para que
    `build_preprocessor` no tenga que conocer los nombres hardcodeados."""

    numeric: list[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))

    @property
    def all_columns(self) -> list[str]:
        return self.numeric + self.categorical


def build_preprocessor(schema: FeatureSchema | None = None) -> ColumnTransformer:
    """Crea el preprocesador (imputación + escalado/encoding) como un
    `ColumnTransformer` de scikit-learn, listo para insertarse en el
    pipeline de modelado (`src/models/train_model.py`).
    """
    schema = schema or FeatureSchema()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, schema.numeric),
            ("categorical", categorical_pipeline, schema.categorical),
        ]
    )


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    schema: FeatureSchema | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa `df` en train/test estratificado por la variable objetivo.

    Devuelve (X_train, X_test, y_train, y_test), donde X_* solo contiene las
    columnas de `schema` (las predictoras), no todas las columnas de `df`.
    """
    schema = schema or FeatureSchema()
    X = df[schema.all_columns]
    y = df[target_column]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

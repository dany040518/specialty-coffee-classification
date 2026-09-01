"""Entrenamiento de modelos (fase 04 - modelado).

Decisiones tomadas para esta iteración (ver `config/config.yaml -> models`):

- **Modelo:** regresión logística como baseline (el notebook original la
  importaba junto con árboles/ensambles, pero nunca llegó a entrenar nada;
  se elige como punto de partida interpretable, dejando `MODEL_REGISTRY`
  abierto para agregar Random Forest / XGBoost en una siguiente iteración
  sin tocar `SklearnModelTrainer`).
- **Balanceo de clases:** SMOTE sobre el conjunto de entrenamiento
  únicamente (el `Pipeline` de `imblearn` evita que el resampling se
  aplique al validar o predecir, para no filtrar información sintética al
  test).
- **Validación:** `StratifiedKFold` sobre train, con la semilla de
  `config.yaml -> project.random_seed`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import resolve_path
from src.features.build_features import FeatureSchema, build_preprocessor
from src.models.base import ModelTrainer

# Registro de modelos disponibles: agregar un modelo nuevo es agregar una
# entrada aquí (y sus hiperparámetros en config.yaml), sin modificar
# SklearnModelTrainer (Open/Closed Principle).
MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
}

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


class SklearnModelTrainer(ModelTrainer):
    """Entrena `preprocesador -> SMOTE -> estimador` como un único pipeline
    de `imblearn`, para que el balanceo de clases nunca se aplique fuera del
    conjunto de entrenamiento.
    """

    def __init__(
        self,
        model_name: str,
        hyperparams: dict[str, Any],
        schema: FeatureSchema | None = None,
        random_seed: int = 42,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Modelo desconocido: {model_name!r}. Disponibles: {list(MODEL_REGISTRY)}"
            )
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.schema = schema or FeatureSchema()
        self.random_seed = random_seed

    def build_pipeline(self) -> ImbPipeline:
        estimator_cls = MODEL_REGISTRY[self.model_name]
        estimator = estimator_cls(random_state=self.random_seed, **self.hyperparams)
        return ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.schema)),
                ("smote", SMOTE(random_state=self.random_seed)),
                ("classifier", estimator),
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> ImbPipeline:
        pipeline = self.build_pipeline()
        pipeline.fit(X, y)
        return pipeline

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict[str, float]:
        """Valida el pipeline (sin ajustarlo de forma definitiva) con
        `StratifiedKFold`, devolviendo el promedio de cada métrica.
        """
        pipeline = self.build_pipeline()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=CV_SCORING)
        return {metric: float(scores[f"test_{metric}"].mean()) for metric in CV_SCORING}


def save_model(pipeline: ImbPipeline, metadata: dict[str, Any], config: dict) -> Path:
    """Serializa el pipeline entrenado en `models/` junto con un JSON de
    metadatos (hiperparámetros, fecha, semilla, métricas de validación).
    """
    import joblib

    models_dir = resolve_path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = metadata.get("model_name", "model")
    model_path = models_dir / f"{model_name}.joblib"
    metadata_path = models_dir / f"{model_name}_metadata.json"

    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return model_path


def train_from_config(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "logistic_regression",
    cv_folds: int = 5,
) -> tuple[ImbPipeline, dict[str, Any]]:
    """Orquesta el entrenamiento: lee hiperparámetros de `config.yaml`,
    valida con CV, entrena el modelo final sobre todo `X_train`/`y_train` y
    lo guarda en `models/`. Devuelve (pipeline_entrenado, metadata).
    """
    random_seed = config["project"]["random_seed"]
    hyperparams = config["models"].get(model_name, {})

    trainer = SklearnModelTrainer(
        model_name=model_name,
        hyperparams=hyperparams,
        random_seed=random_seed,
    )
    cv_metrics = trainer.cross_validate(X_train, y_train, cv_folds=cv_folds)
    pipeline = trainer.train(X_train, y_train)

    metadata = {
        "model_name": model_name,
        "hyperparams": hyperparams,
        "random_seed": random_seed,
        "cv_folds": cv_folds,
        "cv_metrics": cv_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": len(X_train),
    }
    save_model(pipeline, metadata, config)
    return pipeline, metadata

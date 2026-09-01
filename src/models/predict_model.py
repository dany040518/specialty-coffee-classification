"""Generación de predicciones con un modelo ya entrenado (fase 05/06)."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.models.base import Predictor


class ModelPredictor(Predictor):
    """Carga un pipeline serializado (`.joblib`, guardado por
    `train_model.save_model`) y genera predicciones sobre datos nuevos.
    """

    def __init__(self, model_path: Path | str):
        import joblib

        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

    @classmethod
    def from_config(cls, config: dict, model_name: str = "logistic_regression") -> "ModelPredictor":
        models_dir = resolve_path(config["paths"]["models_dir"])
        return cls(models_dir / f"{model_name}.joblib")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

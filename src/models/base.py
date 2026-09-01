"""Interfaces para la capa de modelado."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ModelTrainer(ABC):
    """Entrena un estimador y devuelve el pipeline ya ajustado.

    Nuevos modelos (Random Forest, XGBoost, ...) se agregan registrando un
    nuevo `ModelTrainer` (o una nueva entrada en `MODEL_REGISTRY`, ver
    `train_model.py`), sin modificar el código de orquestación (OCP).
    """

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        raise NotImplementedError


class Predictor(ABC):
    """Genera predicciones con un modelo ya entrenado."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class Evaluator(ABC):
    """Calcula métricas de desempeño para un conjunto de predicciones."""

    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        raise NotImplementedError

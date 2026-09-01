"""Capa de orquestación (fachada) del pipeline CRISP-DM.

Este es el ÚNICO módulo que los notebooks deberían importar para ejecutar
una fase completa. Cada notebook (03 a 06) llama a una función de aquí y se
limita a mostrar/comentar los resultados — así, `notebooks/` queda como
texto + análisis, y toda la lógica vive en `src/` (Dependency Inversion: la
orquestación depende de las interfaces de cada capa, no al revés).

Fases cubiertas:
    prepare_data  -> notebook 03 (preparación de los datos)
    train         -> notebook 04 (modelado)
    evaluate       -> notebook 05 (evaluación)
"""

from typing import Any

import pandas as pd

from src.config import resolve_path
from src.data.clean_data import CleaningPipeline, DropCorruptScores, DropDuplicateRows
from src.data.load_data import CQIRawDataLoader
from src.features.build_features import FeatureSchema, SpecialtyTargetBuilder, split_train_test
from src.models.evaluate_model import ClassificationEvaluator, save_metrics_json
from src.models.predict_model import ModelPredictor
from src.models.train_model import train_from_config
from src.visualization.plots import plot_confusion_matrix, plot_roc_curve

SPECIES = ("arabica", "robusta")


def load_combined_raw(config: dict) -> pd.DataFrame:
    """Carga arabica + robusta y las combina en un único DataFrame, con una
    columna `species` (minúsculas) derivada de la columna `Species` de CQI.
    """
    frames = []
    for species in SPECIES:
        df = CQIRawDataLoader(species, config).load()
        df["species"] = species
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare_data(config: dict) -> dict[str, Any]:
    """Fase 03: limpia los datos, construye el target y separa train/test.

    Guarda `data/processed/train.csv`, `data/processed/test.csv` (features +
    target) y `data/processed/coffee_processed.csv` (dataset completo
    limpio, para EDA/interpretación en 06). Devuelve un dict con los splits
    en memoria y las estadísticas de limpieza, para mostrarlas en el
    notebook 03.
    """
    raw = load_combined_raw(config)

    duplicates_step = DropDuplicateRows()
    corrupt_step = DropCorruptScores(
        score_column=config["target"]["source_column"],
        min_valid_score=0.0,
    )
    cleaned = CleaningPipeline([duplicates_step, corrupt_step]).run(raw)

    target_builder = SpecialtyTargetBuilder.from_config(config)
    processed = target_builder.build(cleaned)

    schema = FeatureSchema()
    X_train, X_test, y_train, y_test = split_train_test(
        processed,
        target_column=config["target"]["target_column"],
        schema=schema,
        test_size=config["split"]["test_size"],
        random_seed=config["project"]["random_seed"],
    )

    processed_dir = resolve_path(config["paths"]["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_dir / "coffee_processed.csv", index=False)
    pd.concat([X_train, y_train], axis=1).to_csv(processed_dir / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(processed_dir / "test.csv", index=False)

    return {
        "raw": raw,
        "processed": processed,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_duplicates_dropped": duplicates_step.n_dropped_,
        "n_corrupt_dropped": corrupt_step.n_dropped_,
    }


def load_processed_split(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Recarga train/test ya guardados por `prepare_data`, para notebooks
    (04, 05) que se ejecutan de forma independiente al 03.
    """
    processed_dir = resolve_path(config["paths"]["data"]["processed_dir"])
    target_column = config["target"]["target_column"]
    schema = FeatureSchema()

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    return (
        train_df[schema.all_columns],
        test_df[schema.all_columns],
        train_df[target_column],
        test_df[target_column],
    )


def train(config: dict, model_name: str = "logistic_regression", cv_folds: int = 5) -> dict[str, Any]:
    """Fase 04: entrena el modelo (con CV) sobre `data/processed/train.csv`
    y lo guarda en `models/`.
    """
    X_train, _, y_train, _ = load_processed_split(config)
    pipeline, metadata = train_from_config(
        config, X_train, y_train, model_name=model_name, cv_folds=cv_folds
    )
    return {"pipeline": pipeline, "metadata": metadata, "X_train": X_train, "y_train": y_train}


def evaluate(config: dict, model_name: str = "logistic_regression") -> dict[str, Any]:
    """Fase 05: evalúa el modelo guardado sobre `data/processed/test.csv`,
    guarda las métricas en `reports/tables/` y las gráficas en
    `reports/figures/`.
    """
    _, X_test, _, y_test = load_processed_split(config)

    predictor = ModelPredictor.from_config(config, model_name=model_name)
    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    cm = evaluator.confusion_matrix(y_test, y_pred)
    fpr, tpr = evaluator.roc_curve_points(y_test, y_proba)

    save_metrics_json(metrics, config, filename=f"{model_name}_metrics.json")
    plot_confusion_matrix(cm, config, filename=f"{model_name}_confusion_matrix.png")
    plot_roc_curve(fpr, tpr, metrics["roc_auc"], config, filename=f"{model_name}_roc_curve.png")

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

"""Обёртка над MLflow: запуск, логирование параметров, модели и метрик.

Если переменная MLFLOW_TRACKING_URI не задана, все функции — no-op.
Это позволяет запускать пайплайн локально без MLflow.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False

def _enabled() -> bool:
    return MLFLOW_AVAILABLE and bool(os.environ.get("MLFLOW_TRACKING_URI"))

def start_run(experiment: str, params: dict[str, Any]) -> None:
    if not _enabled():
        return
    mlflow.set_experiment(experiment)
    mlflow.start_run()
    mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))})

def log_model(model, feature_names: list[str], sample: pd.DataFrame) -> None:
    if not _enabled():
        return
    try:
        signature = infer_signature(sample, model.predict(sample))
        mlflow.sklearn.log_model(model, "model", signature=signature)
    except Exception:  # pragma: no cover
        mlflow.sklearn.log_model(model, "model")

def log_metrics(metrics: dict[str, float]) -> None:
    if not _enabled():
        return
    mlflow.log_metrics(metrics)

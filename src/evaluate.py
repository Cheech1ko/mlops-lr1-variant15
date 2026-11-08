"""Этап evaluate: метрики классификации на тесте, запись metrics.json.

Основная метрика — recall_macro (усреднённый по классам recall).
Если recall_macro ниже порога evaluate.min_recall, этап завершается с ошибкой.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import tracking
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("evaluate")

FEATURES = Path("data/features/test.csv")
MODEL = Path("models/model.joblib")
METRICS_OUT = Path("reports/metrics.json")
PLOT_OUT = Path("reports/plots/confusion_matrix.csv")


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    min_recall = float(params["evaluate"]["min_recall"])

    bundle = joblib.load(MODEL)
    model, feature_names = bundle["model"], bundle["features"]

    frame = pd.read_csv(FEATURES)
    x_test = frame[feature_names]
    y_test = frame[target]
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(
            float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        "n_test": int(len(frame)),
    }
    
    # Теперь ensure_parent правильно создает папку reports/
    ensure_parent(METRICS_OUT).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    ensure_parent(PLOT_OUT.parent).mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(PLOT_OUT)

    log.info("classification_report:\n%s", classification_report(y_test, y_pred, zero_division=0))
    tracking.log_metrics({k: v for k, v in metrics.items() if k != "n_test"})
    log.info("метрики: %s", metrics)

    if metrics["recall_macro"] < min_recall:
        log.error(
            "recall_macro = %.4f ниже порога %.2f",
            metrics["recall_macro"],
            min_recall,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

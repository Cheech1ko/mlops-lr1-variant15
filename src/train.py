"""Этап train: обучение классификатора на подготовленных признаках."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from src import tracking
from src.utils import ensure_parent, get_logger, load_params, set_seed

log = get_logger("train")

FEATURES = Path("data/features/train.csv")
MODEL_OUT = Path("models/model.joblib")

def make_model(train_params: dict):
    name = train_params["model"]
    seed = int(train_params["seed"])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(train_params["n_estimators"]),
            max_depth=int(train_params["max_depth"]),
            min_samples_leaf=int(train_params["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(train_params["n_estimators"]),
            max_depth=int(train_params["max_depth"]),
            min_samples_leaf=int(train_params["min_samples_leaf"]),
            learning_rate=float(train_params["learning_rate"]),
            random_state=seed,
        )
    raise ValueError(f"неизвестная модель: {name}")

def main() -> None:
    params = load_params()
    train_params = params["train"]
    target = params["data"]["target"]
    set_seed(int(train_params["seed"]))

    frame = pd.read_csv(FEATURES)
    x_train = frame.drop(columns=[target])
    y_train = frame[target]

    tracking.start_run(params["mlflow"]["experiment"], dict(train_params))

    model = make_model(train_params)
    model.fit(x_train, y_train)

    bundle = {"model": model, "features": list(x_train.columns)}
    joblib.dump(bundle, ensure_parent(MODEL_OUT))
    tracking.log_model(model, list(x_train.columns), x_train.head(50))

    log.info(
        "обучено: %s на %d строках -> %s", train_params["model"], len(frame), MODEL_OUT
    )

if __name__ == "__main__":
    main()

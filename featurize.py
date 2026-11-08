"""Этап featurize: построение sklearn Pipeline для признаков.

Все преобразования (импутация, one-hot) инкапсулированы в Pipeline,
который сохраняется в models/preprocessor.joblib. Это гарантирует,
что при инференсе применяются те же преобразования, что и при обучении.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("featurize")

IN = {
    "train": Path("data/processed/train.csv"),
    "test": Path("data/processed/test.csv"),
}
OUT = {
    "train": Path("data/features/train.csv"),
    "test": Path("data/features/test.csv"),
}
PREPROCESSOR_OUT = Path("models/preprocessor.joblib")

def build_preprocessor(
    frame: pd.DataFrame,
    target: str,
    drop: list[str],
    numeric_imputer_strategy: str,
    categorical_imputer_strategy: str,
    onehot_handle_unknown: str,
    onehot_min_frequency: int,
) -> tuple[Pipeline, list[str], list[str]]:
    """Собирает ColumnTransformer: числовые — median, категориальные — most_frequent + OHE."""
    feature_frame = frame.drop(columns=[c for c in drop if c in frame.columns], errors="ignore")
    feature_frame = feature_frame.drop(columns=[target], errors="ignore")

    numeric_cols = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=numeric_imputer_strategy)),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=categorical_imputer_strategy)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=onehot_handle_unknown,
                    min_frequency=onehot_min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols

def main() -> None:
    params = load_params()
    data_params = params["data"]
    feat_params = params["featurize"]
    target = data_params["target"]
    drop = list(data_params["drop"])

    train = pd.read_csv(IN["train"])
    test = pd.read_csv(IN["test"])

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        train,
        target=target,
        drop=drop,
        numeric_imputer_strategy=feat_params["numeric_imputer_strategy"],
        categorical_imputer_strategy=feat_params["categorical_imputer_strategy"],
        onehot_handle_unknown=feat_params["onehot_handle_unknown"],
        onehot_min_frequency=int(feat_params["onehot_min_frequency"]),
    )

    x_train = preprocessor.fit_transform(train)
    x_test = preprocessor.transform(test)

    feature_names = preprocessor.get_feature_names_out()
    train_out = pd.DataFrame(x_train, columns=feature_names)
    train_out[target] = train[target].values
    test_out = pd.DataFrame(x_test, columns=feature_names)
    test_out[target] = test[target].values

    train_out.to_csv(ensure_parent(OUT["train"]), index=False)
    test_out.to_csv(ensure_parent(OUT["test"]), index=False)
    joblib.dump(preprocessor, ensure_parent(PREPROCESSOR_OUT))

    log.info(
        "train: %d строк, %d признаков (num=%d, cat=%d -> OHE)",
        len(train_out),
        train_out.shape[1] - 1,
        len(numeric_cols),
        len(categorical_cols),
    )
    log.info("test: %d строк, %d признаков", len(test_out), test_out.shape[1] - 1)
    log.info("препроцессор сохранён: %s", PREPROCESSOR_OUT)

if __name__ == "__main__":
    main()

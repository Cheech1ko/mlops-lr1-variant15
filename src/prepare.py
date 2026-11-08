"""Этап prepare: проверка данных и разбиение на обучение и тест.

Разбиение по возрасту: train — пациенты с age <= age_threshold,
test — пациенты с age > age_threshold. Это совпадает с логикой потока
для ЛР4 (пациенты до 60 лет — обучение, старше — поток).

Случайное разбиение (split_by_age: false) даёт завышенную метрику
из-за утечки: пациенты одного возраста попадают и в train, и в test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schema import validate_raw
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("prepare")

TRAIN_OUT = Path("data/processed/train.csv")
TEST_OUT = Path("data/processed/test.csv")

def age_to_midpoint(age: str) -> float:
    """Преобразует строковый диапазон '[50-60)' в середину 55.0."""
    # Убираем скобки и заменяем возможные тире на дефис
    clean_age = str(age).replace("[", "").replace(")", "").replace("–", "-").replace("—", "-")
    lo, hi = clean_age.split("-")
    return (float(lo) + float(hi)) / 2.0
    
def check(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    validated = validate_raw(frame)
    log.info("схема пройдена: %d строк, %d колонок", len(validated), validated.shape[1])
    if validated[target].isna().any():
        raise ValueError(f"в целевой переменной {target} есть пропуски")
    return validated

def split(
    frame: pd.DataFrame,
    age_threshold: float,
    by_age: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if by_age:
        # Строгое приведение к float, чтобы избежать проблем с типами
        age_mid = frame["age"].apply(age_to_midpoint).astype(float)
        train = frame[age_mid <= float(age_threshold)].copy()
        test = frame[age_mid > float(age_threshold)].copy()
        return train, test
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    edge = int(len(shuffled) * 0.8)
    return shuffled.iloc[:edge].copy(), shuffled.iloc[edge:].copy()

def main() -> None:
    params = load_params()
    data_params = params["data"]
    target = data_params["target"]

    raw = pd.read_csv(data_params["raw"])
    raw = check(raw, target)

    train, test = split(
        raw,
        age_threshold=float(data_params["age_threshold"]),
        by_age=bool(data_params["split_by_age"]),
        seed=int(params["train"]["seed"]),
    )

    if train.empty or test.empty:
        raise ValueError("после разбиения train или test пуст")

    train.to_csv(ensure_parent(TRAIN_OUT), index=False)
    test.to_csv(ensure_parent(TEST_OUT), index=False)
    log.info(
        "train: %d строк (age <= %s), test: %d строк (age > %s)",
        len(train),
        data_params["age_threshold"],
        len(test),
        data_params["age_threshold"],
    )

if __name__ == "__main__":
    main()

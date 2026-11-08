from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schema import validate_raw
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("prepare")

TRAIN_OUT = Path("data/processed/train.csv")
TEST_OUT = Path("data/processed/test.csv")


def parse_age(age_str: str) -> float:
    """Надежно вытаскивает числа из строки возраста, игнорируя скобки и тире."""
    import re
    # Находим все числа в строке (например, из "[50-60)" достанет 50 и 60)
    numbers = re.findall(r'\d+', str(age_str))
    if len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2.0
    elif len(numbers) == 1:
        return float(numbers[0])
    return 0.0


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
        # Применяем надежный парсер к каждому возрасту
        age_mid = frame["age"].apply(parse_age)
        
        train = frame[age_mid <= float(age_threshold)].copy()
        test = frame[age_mid > float(age_threshold)].copy()
        
        # ДИАГНОСТИКА: выведем в консоль, сколько строк получилось
        log.info("Всего строк: %d. Разделение по возрасту (порог %s):", len(frame), age_threshold)
        log.info("  - train: %d строк", len(train))
        log.info("  - test: %d строк", len(test))
        
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
        # Если тест пуст, значит все пациенты младше 60 лет, либо парсер не сработал
        raise ValueError(f"После разбиения train или test пуст. Train: {len(train)}, Test: {len(test)}")

    train.to_csv(ensure_parent(TRAIN_OUT), index=False)
    test.to_csv(ensure_parent(TEST_OUT), index=False)
    log.info("Файлы train.csv и test.csv успешно сохранены!")


if __name__ == "__main__":
    main()

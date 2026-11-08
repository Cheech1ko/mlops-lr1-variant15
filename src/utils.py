"""Общие вспомогательные функции: параметры, журналирование, фиксация случайности."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PARAMS_PATH = Path("params.yaml")

def load_params(path: Path | str = PARAMS_PATH) -> dict[str, Any]:
    """Читает params.yaml. safe_load не исполняет произвольные теги YAML."""
    text = Path(path).read_text(encoding="utf-8")
    params = yaml.safe_load(text)
    if not isinstance(params, dict):
        raise TypeError(f"{path}: ожидался словарь параметров")
    return params

def set_seed(seed: int) -> np.random.Generator:
    """Фиксирует случайность и возвращает генератор NumPy для явной передачи."""
    random.seed(seed)
    return np.random.default_rng(seed)

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

def ensure_parent(path: Path) -> Path:
    """Создаёт каталог для файла, если его нет, и возвращает путь."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

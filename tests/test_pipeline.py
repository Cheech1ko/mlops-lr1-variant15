"""Базовые тесты пайплайна: параметры загружаются, разбиение корректно."""

from __future__ import annotations

import pandas as pd
import pytest

from src.prepare import age_to_midpoint, split
from src.utils import load_params

def test_load_params():
    params = load_params()
    assert "data" in params
    assert params["data"]["target"] == "readmitted"

def test_age_to_midpoint():
    assert age_to_midpoint("[0-10)") == 5.0
    assert age_to_midpoint("[50-60)") == 55.0
    assert age_to_midpoint("[90-100)") == 95.0

def test_split_by_age():
    df = pd.DataFrame({
        "age": ["[0-10)", "[50-60)", "[70-80)", "[80-90)"],
        "readmitted": ["NO", ">30", "<30", "NO"],
    })
    train, test = split(df, age_threshold=60, by_age=True, seed=42)
    assert len(train) == 2
    assert len(test) == 2
    assert all(age_to_midpoint(a) <= 60 for a in train["age"])
    assert all(age_to_midpoint(a) > 60 for a in test["age"])

def test_split_random():
    df = pd.DataFrame({
        "age": ["[0-10)", "[50-60)", "[70-80)", "[80-90)"] * 25,
        "readmitted": ["NO", ">30", "<30", "NO"] * 25,
    })
    train, test = split(df, age_threshold=60, by_age=False, seed=42)
    assert len(train) == 80
    assert len(test) == 20

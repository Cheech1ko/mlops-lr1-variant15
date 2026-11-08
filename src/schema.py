"""Схема проверки исходных данных Diabetes 130-US Hospitals (pandera)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

RAW_SCHEMA = DataFrameSchema(
    {
        "encounter_id": Column(int, nullable=False),
        "patient_nbr": Column(int, nullable=False),
        "race": Column(str, nullable=True),
        "gender": Column(str, nullable=False),
        "age": Column(str, nullable=False),
        "readmitted": Column(str, nullable=False),
    },
    strict=False,
    coerce=False,
)

def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Проверяет обязательные колонки и типы. Остальные колонки не трогает."""
    try:
        return RAW_SCHEMA.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(f"схема исходных данных не пройдена:\n{exc}") from exc

# Лабораторная работа 1. Воспроизводимый проект

**Вариант 15:** Diabetes 130-US Hospitals (UCI)
**Задача:** Многоклассовая классификация повторной госпитализации (`readmitted`: `<30`, `>30`, `NO`)
**Поток для ЛР4:** Пациенты до 60 лет — обучение, старше 60 — поток.

## 1. Постановка задачи
Цель работы — собрать воспроизводимый ML-проект с версионированием данных и пайплайном обработки. Данные взяты из UCI ML Repository. В качестве целевой переменной используется `readmitted`, разделенная на 3 класса. Разбиение на train/test производится по возрасту: тест — пациенты старше 60 лет.

## 2. Структура проекта
- `src/` — код пайплайна (`ingest.py`, `prepare.py`, `featurize.py`, `train.py`, `evaluate.py`, `tracking.py`).
- `notebooks/` — рабочий ноутбук `lr1_data_and_pipeline.ipynb`.
- `tests/` — базовые тесты.
- `data/` — данные (raw под DVC, processed и features генерируются).
- `models/` — сохраненная модель и препроцессор (sklearn Pipeline).
- `reports/` — метрики и графики.
- `params.yaml` — все параметры пайплайна.
- `dvc.yaml` — описание этапов DVC.

## 3. Развертывание окружения (Colab)
Окружение фиксируется файлами `pyproject.toml` и `uv.lock`.
```bash
pip install -q uv
uv export --frozen --no-hashes -o /content/requirements.lock.txt
pip install -q -r /content/requirements.lock.txt

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("ingest")


def download(url: str, dest: Path) -> Path:
    """Скачивает файл по URL, если его ещё нет."""
    if dest.exists():
        log.info("архив уже скачан: %s", dest)
        return dest
    ensure_parent(dest)
    log.info("скачивание %s -> %s", url, dest)
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())
    return dest


def extract_csv(archive: Path, csv_path_in_zip: str, dest: Path) -> Path:
    """Распаковывает указанный CSV из zip-архива."""
    if dest.exists():
        log.info("CSV уже распакован: %s", dest)
        return dest
    ensure_parent(dest)
    with zipfile.ZipFile(archive) as zf:
        # Читаем файл и сразу пишем его в нужное место, обходя ошибку с путями
        with zf.open(csv_path_in_zip) as src:
            dest.write_bytes(src.read())
    log.info("распакован %s -> %s", csv_path_in_zip, dest)
    return dest


def main() -> None:
    params = load_params()
    ing = params["ingest"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--age-filter",
        nargs="*",
        default=ing.get("age_filter"),
        help="список возрастных диапазонов для v1; без флага — полный набор",
    )
    parser.add_argument("--source-url", default=ing["source_url"])
    parser.add_argument("--out", type=Path, default=Path(ing["output"]))
    args = parser.parse_args()

    archive = download(args.source_url, Path(ing["raw_archive"]))
    # Указываем правильный путь к файлу внутри архива
    csv_path = extract_csv(archive, "dataset_diabetes/diabetic_data.csv", Path(ing["raw_csv"]))

    df = pd.read_csv(csv_path)
    log.info("исходный набор: %d строк, %d колонок", len(df), df.shape[1])

    if args.age_filter:
        before = len(df)
        df = df[df["age"].isin(args.age_filter)].copy()
        log.info(
            "фильтр по возрасту %s: %d -> %d строк",
            args.age_filter,
            before,
            len(df),
        )

    if df.empty:
        raise ValueError("после фильтрации не осталось строк")

    df.to_csv(ensure_parent(args.out), index=False)
    log.info("сохранено: %s строк -> %s", len(df), args.out)


if __name__ == "__main__":
    main()

"""Validate the standalone public-release candidate.

Checks that commonly regress when results are copied out of a local experiment tree:
relative data paths, patient/split consistency, cleared notebook outputs, and absence of
Windows user paths in text files. Run from any directory with
``python guards/check_release.py``.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOCAL_PATH = re.compile(
    r"(?:[A-Za-z]:\\(?:Users|Kuliah|Documents|Desktop|Downloads|OneDrive)\\"
    r"|\\Users\\|/Users/|/home/[^/]+/)"
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".cff", ".ipynb"}
PATH_COLUMNS = ("img_path", "li_path", "ma_path", "rect_path")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_index(errors: list[str]) -> set[str]:
    path = DATA / "master_index.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2676:
        fail(errors, f"master_index.csv has {len(rows)} rows; expected 2676")
    required = {"image_id", "patient_id", "release", "center", *PATH_COLUMNS}
    missing = required - set(rows[0]) if rows else required
    if missing:
        fail(errors, f"master_index.csv is missing columns: {sorted(missing)}")
        return set()
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        fail(errors, "master_index.csv contains duplicate image_id values")
    for line_no, row in enumerate(rows, 2):
        for column in PATH_COLUMNS:
            value = row[column]
            if value and (Path(value).is_absolute() or LOCAL_PATH.search(value)):
                fail(errors, f"master_index.csv:{line_no} has non-portable {column}")
    return {row["patient_id"] for row in rows}


def check_splits(errors: list[str], patient_ids: set[str]) -> None:
    with (DATA / "splits_5fold.json").open(encoding="utf-8") as handle:
        splits = json.load(handle)
    test = set(splits.get("test", []))
    folds = splits.get("folds", {})
    if len(folds) != 5:
        fail(errors, f"splits_5fold.json has {len(folds)} folds; expected 5")
    if not test:
        fail(errors, "splits_5fold.json has an empty fixed test set")
    validation_sets = {name: set(ids) for name, ids in folds.items()}
    development = set().union(*validation_sets.values()) if validation_sets else set()
    if development & test:
        fail(errors, "development folds overlap the fixed test set")
    if set().union(test, development) != patient_ids:
        fail(errors, "test and validation-fold assignments do not cover all patients")
    assigned = sum(len(ids) for ids in validation_sets.values())
    if assigned != len(development):
        fail(errors, "a patient occurs in more than one validation fold")
    for fold_name, val in validation_sets.items():
        train = development - val
        overlap = (train & val) | (train & test) | (val & test)
        if overlap:
            fail(errors, f"{fold_name} is not patient-disjoint")
        unknown = (train | val | test) - patient_ids
        if unknown:
            fail(errors, f"{fold_name} references {len(unknown)} unknown patients")

    with (DATA / "splits_loco.json").open(encoding="utf-8") as handle:
        loco = json.load(handle)
    expected = {"Cyprus", "Munich", "Pisa_clin", "Pisa_tech", "Porto", "Torino", "Toronto"}
    if set(loco) != expected:
        fail(errors, "splits_loco.json does not contain the seven expected source strata")
    for source, counts in loco.items():
        if counts.get("n_train", 0) + counts.get("n_test", 0) != 2676:
            fail(errors, f"LOCO counts for {source} do not total 2676")


def check_notebooks(errors: list[str]) -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        with path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)
        outputs = sum(len(cell.get("outputs", [])) for cell in notebook.get("cells", []))
        counts = [cell.get("execution_count") for cell in notebook.get("cells", [])]
        if outputs or any(count is not None for count in counts):
            fail(errors, f"{path.relative_to(ROOT)} still contains execution output or counts")


def check_local_paths(errors: list[str]) -> None:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if LOCAL_PATH.search(text):
            fail(errors, f"{path.relative_to(ROOT)} contains a local absolute path")


def main() -> int:
    errors: list[str] = []
    patient_ids = check_index(errors)
    check_splits(errors, patient_ids)
    check_notebooks(errors)
    check_local_paths(errors)
    if errors:
        print(f"release check failed ({len(errors)} finding(s)):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("release check passed: index, splits, notebooks, and paths are portable")
    return 0


if __name__ == "__main__":
    sys.exit(main())

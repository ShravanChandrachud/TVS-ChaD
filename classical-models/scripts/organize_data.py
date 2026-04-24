"""
Data Organizer — Unify All Raw Data into Train/Test/Val Structure
==================================================================
Takes the messy data/raw/ directory (where datasets arrived in various
formats — loose images, pre-split folders with CSVs, nested paths like
food-101/images/omelette, classification subfolders) and produces a
clean unified structure:

    data/organized/<object>/<state>/train/
    data/organized/<object>/<state>/test/
    data/organized/<object>/<state>/val/

Handles five source formats automatically:
    1. "loose"          — images directly in folder, auto-split 80/10/10
    2. "splits"         — folder already has train/test/val with images
    3. "splits_csv"     — train/test/val with _classes.csv listing filenames
    4. "nested"         — images buried in subfolders (e.g. food-101/images/omelette)
    5. "classification" — images in a named subfolder (e.g. Eggs classification/Not Damaged)

All source-to-class mappings are defined in the SOURCES list at the top
of the script — edit that list if folder names or paths change.

Input:  data/raw/ (multiple formats)
Output: data/organized/<object>/<state>/(train|test|val)/*.jpg
"""

import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ORG_DIR = PROJECT_ROOT / "data" / "organized"

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
TEST_RATIO = 0.10
VAL_RATIO = 0.10

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SOURCES = [
    {
        "object": "bowl",
        "state": "empty",
        "sources": [
            (
                "splits_csv",
                "bowl/empty/non steel",
                None,
            ),
            ("splits_csv", "bowl/empty/steel bowls", None),
        ],
    },
    {
        "object": "bowl",
        "state": "with-whisked-egg",
        "sources": [
            ("loose", "bowl/with-whisked-egg", None),
        ],
    },
    {
        "object": "butter",
        "state": "solid",
        "sources": [
            ("splits_csv", "butter/solid", None),
        ],
    },
    {
        "object": "butter",
        "state": "melted",
        "sources": [
            ("loose", "butter/melted", None),
        ],
    },
    {
        "object": "eggs",
        "state": "burned",
        "sources": [
            ("loose", "eggs/burned", None),
        ],
    },
    {
        "object": "eggs",
        "state": "omelette",
        "sources": [
            ("nested", "eggs/omelette", "food-101/images/omelette"),
            ("splits_csv", "eggs/omelette/white-omelette", None),
            ("splits_csv", "eggs/omelette/yellow-omelette", None),
        ],
    },
    {
        "object": "eggs",
        "state": "raw",
        "sources": [
            ("classification", "eggs/raw", "Eggs classification/Not Damaged"),
            ("splits", "eggs/raw", None),
        ],
    },
    {
        "object": "eggs",
        "state": "whisked",
        "sources": [
            ("loose", "eggs/whisked", None),
        ],
    },
    {
        "object": "oil",
        "state": "general",
        "sources": [
            ("splits_csv", "oil", None),
        ],
    },
    {
        "object": "pan",
        "state": "empty",
        "sources": [
            ("loose", "pan/empty", None),
        ],
    },
    {
        "object": "pan",
        "state": "with-food",
        "sources": [
            ("loose", "pan/with-food", None),
        ],
    },
    {
        "object": "pepper",
        "state": "ground",
        "sources": [
            ("loose", "pepper/black pepper", None),
        ],
    },
    {
        "object": "plate",
        "state": "empty",
        "sources": [
            ("loose", "plate/empty", None),
        ],
    },
    {
        "object": "plate",
        "state": "with-food",
        "sources": [
            ("loose", "plate/with-food", None),
        ],
    },
]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def get_loose_images(folder: Path) -> list:
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if is_image(f)]


def get_all_images_recursive(folder: Path) -> list:
    if not folder.exists():
        return []
    return [f for f in folder.rglob("*") if is_image(f)]


def split_images(images: list) -> dict:
    random.seed(RANDOM_SEED)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_test = int(n * TEST_RATIO)

    return {
        "train": shuffled[:n_train],
        "test": shuffled[n_train : n_train + n_test],
        "val": shuffled[n_train + n_test :],
    }


def copy_images(images: list, dest_dir: Path, prefix: str) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in images:
        new_name = f"{prefix}_{img.name}" if prefix else img.name
        dest_path = dest_dir / new_name
        if not dest_path.exists():
            shutil.copy2(img, dest_path)
        count += 1
    return count


def process_loose(raw_path: Path, dest_base: Path, prefix: str) -> dict:
    images = get_loose_images(raw_path)
    if not images:
        return {"train": 0, "test": 0, "val": 0}

    splits = split_images(images)
    counts = {}
    for split_name, split_imgs in splits.items():
        counts[split_name] = copy_images(split_imgs, dest_base / split_name, prefix)
    return counts


def process_splits(raw_path: Path, dest_base: Path, prefix: str) -> dict:
    counts = {}
    for split_name in ["train", "test", "val", "valid"]:
        split_dir = raw_path / split_name
        if not split_dir.exists():
            continue

        out_name = "val" if split_name == "valid" else split_name

        images = get_loose_images(split_dir)
        if images:
            counts[out_name] = copy_images(images, dest_base / out_name, prefix)

    return counts


def process_splits_csv(
    raw_path: Path, dest_base: Path, prefix: str, filter_col: str = None
) -> dict:
    counts = {}
    for split_name in ["train", "test", "val", "valid"]:
        split_dir = raw_path / split_name
        csv_path = split_dir / "_classes.csv"

        if not split_dir.exists():
            continue

        out_name = "val" if split_name == "valid" else split_name

        if csv_path.exists():
            valid_files = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("filename", "")
                    if filter_col:
                        if row.get(filter_col, "0").strip() == "1":
                            img_path = split_dir / filename
                            if img_path.exists():
                                valid_files.append(img_path)
                    else:
                        img_path = split_dir / filename
                        if img_path.exists():
                            valid_files.append(img_path)

            if valid_files:
                counts[out_name] = copy_images(
                    valid_files, dest_base / out_name, prefix
                )
        else:
            images = get_loose_images(split_dir)
            if images:
                counts[out_name] = copy_images(images, dest_base / out_name, prefix)

    return counts


def process_nested(
    raw_path: Path, dest_base: Path, prefix: str, nested_path: str
) -> dict:
    full_path = raw_path / nested_path
    images = get_all_images_recursive(full_path)
    if not images:
        return {"train": 0, "test": 0, "val": 0}

    splits = split_images(images)
    counts = {}
    for split_name, split_imgs in splits.items():
        counts[split_name] = copy_images(split_imgs, dest_base / split_name, prefix)
    return counts


def process_classification(
    raw_path: Path, dest_base: Path, prefix: str, subfolder: str
) -> dict:
    full_path = raw_path / subfolder
    images = get_all_images_recursive(full_path)
    if not images:
        return {"train": 0, "test": 0, "val": 0}

    splits = split_images(images)
    counts = {}
    for split_name, split_imgs in splits.items():
        counts[split_name] = copy_images(split_imgs, dest_base / split_name, prefix)
    return counts


def main():
    random.seed(RANDOM_SEED)

    print("\nOrganizing raw data\n")

    summary = {}

    for entry in SOURCES:
        obj = entry["object"]
        state = entry["state"]
        class_label = f"{obj}/{state}"
        dest_base = ORG_DIR / obj / state

        print(f"\n{'─' * 50}")
        print(f"Class: {class_label}")

        total_counts = defaultdict(int)

        for i, (source_type, rel_path, extra) in enumerate(entry["sources"]):
            raw_path = RAW_DIR / rel_path
            prefix = f"src{i}"

            if not raw_path.exists():
                print(f"   ⚠ Not found: {raw_path}, skipping.")
                continue

            print(f"   Source {i}: [{source_type}] {rel_path}")

            if source_type == "loose":
                counts = process_loose(raw_path, dest_base, prefix)
            elif source_type == "splits":
                counts = process_splits(raw_path, dest_base, prefix)
            elif source_type == "splits_csv":
                counts = process_splits_csv(
                    raw_path, dest_base, prefix, filter_col=extra
                )
            elif source_type == "nested":
                counts = process_nested(raw_path, dest_base, prefix, extra)
            elif source_type == "classification":
                counts = process_classification(raw_path, dest_base, prefix, extra)
            else:
                print(f" Unknown source type: {source_type}")
                continue

            for split, count in counts.items():
                total_counts[split] += count
                print(f"      {split}: {count}")

        summary[class_label] = dict(total_counts)

    print("\nORGANIZATION COMPLETE\n")
    print(f"{'Class':<30} {'Train':>7} {'Test':>7} {'Val':>7} {'Total':>7}\n")

    grand_total = 0
    for class_label, counts in summary.items():
        tr = counts.get("train", 0)
        te = counts.get("test", 0)
        va = counts.get("val", 0)
        total = tr + te + va
        grand_total += total
        print(f"{class_label:<30} {tr:>7} {te:>7} {va:>7} {total:>7}")

    print(f"\n{'GRAND TOTAL':<30} {'':>7} {'':>7} {'':>7} {grand_total:>7}")

    print(f"\nOutput: {ORG_DIR}")


if __name__ == "__main__":
    main()

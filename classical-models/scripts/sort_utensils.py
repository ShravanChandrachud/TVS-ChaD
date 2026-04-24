"""
Utensils Sorter — Extract Pan and Plate from Roboflow Kitchenware Dataset
==========================================================================
Reads _classes.csv from each split (train/test/val) in the locally
downloaded kitchenware dataset (data/raw/utensils/) to identify
frying pan and plate images, then copies them into the correct
data/raw/ class folders.

The CSV uses multi-label columns (frying pan, kettle, knife, plate, spoon)
with 0/1 values indicating which objects appear in each image. This script
filters for rows where "frying pan"=1 or "plate"=1 and copies those to:
    data/raw/pan/empty/
    data/raw/plate/empty/

Files are prefixed with the split name to avoid name collisions
across train/test/val.

Input:  data/raw/utensils/(train|test|val)/_classes.csv + images
Output: data/raw/pan/empty/*.jpg, data/raw/plate/empty/*.jpg
"""

import csv
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HF_CACHE = PROJECT_ROOT / "data" / "raw" / "utensils"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MAPPING = {
    "frying pan": ("pan", "empty"),
    "plate": ("plate", "empty"),
}

SPLITS = ["train", "test", "val"]
TARGET_SIZE = (512, 512)


def main():
    if not HF_CACHE.exists():
        print(f"ERROR: {HF_CACHE} not found.")
        return

    print("=" * 60)
    print("Sorting utensils using _classes.csv")
    print(f"Source: {HF_CACHE}")
    print("=" * 60)

    # Create output dirs
    for col_name, (obj, state) in MAPPING.items():
        (RAW_DIR / obj / state).mkdir(parents=True, exist_ok=True)

    stats = {col_name: 0 for col_name in MAPPING}

    for split in SPLITS:
        split_dir = HF_CACHE / split
        csv_path = split_dir / "_classes.csv"

        if not csv_path.exists():
            print(f"\n⚠ No _classes.csv in {split_dir}, skipping.")
            continue

        print(f"\nProcessing: {split} ({csv_path})")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                filename = row["filename"]
                source_path = split_dir / filename

                if not source_path.exists():
                    continue

                for col_name, (obj, state) in MAPPING.items():
                    if row.get(col_name, "0").strip() == "1":
                        dest_dir = RAW_DIR / obj / state
                        new_name = f"{obj}_{split}_{filename}"
                        dest_path = dest_dir / new_name

                        if dest_path.exists():
                            stats[col_name] += 1
                            continue

                        shutil.copy2(source_path, dest_path)
                        stats[col_name] += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    for col_name, count in stats.items():
        obj, state = MAPPING[col_name]
        final = len(list((RAW_DIR / obj / state).glob("*")))
        print(
            f"   {col_name} → {obj}/{state}: {count} copied ({final} total in folder)"
        )

    print(f"\nTotal: {sum(stats.values())}")


if __name__ == "__main__":
    main()

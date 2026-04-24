import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
VAL_DIR = PROJECT_ROOT / "data" / "val"

TRAIN_RATIO = 0.80
TEST_RATIO = 0.10
VAL_RATIO = 0.10

RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    assert abs(TRAIN_RATIO + TEST_RATIO + VAL_RATIO - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    if not RAW_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        print("Run scrape_pexels.py first.")
        return

    if TRAIN_DIR.exists() or TEST_DIR.exists() or VAL_DIR.exists():
        print("⚠ Split directories already exist:")
        for d in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
            if d.exists():
                print(f"   {d}")
        ans = input("Delete and re-split? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        for d in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
            if d.exists():
                shutil.rmtree(d)
                print(f"   Deleted {d}")

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Dataset Splitter")
    print(f"Source:      {RAW_DIR}")
    print(
        f"Split:       {TRAIN_RATIO:.0%} train / {TEST_RATIO:.0%} test / {VAL_RATIO:.0%} val"
    )
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 60)

    stats = defaultdict(lambda: {"total": 0, "train": 0, "test": 0, "val": 0})
    total_stats = {"total": 0, "train": 0, "test": 0, "val": 0}

    for object_dir in sorted(RAW_DIR.iterdir()):
        if not object_dir.is_dir():
            continue
        for state_dir in sorted(object_dir.iterdir()):
            if not state_dir.is_dir():
                continue

            class_label = f"{object_dir.name}/{state_dir.name}"

            images = sorted(
                [f for f in state_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
            )

            if not images:
                print(f"  No images found for {class_label}, skipping.")
                continue

            random.shuffle(images)

            n = len(images)
            n_train = int(n * TRAIN_RATIO)
            n_test = int(n * TEST_RATIO)
            n_val = n - n_train - n_test

            train_imgs = images[:n_train]
            test_imgs = images[n_train : n_train + n_test]
            val_imgs = images[n_train + n_test :]

            splits = {
                "train": (TRAIN_DIR, train_imgs),
                "test": (TEST_DIR, test_imgs),
                "val": (VAL_DIR, val_imgs),
            }

            for split_name, (split_dir, split_imgs) in splits.items():
                dest = split_dir / object_dir.name / state_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for img_path in split_imgs:
                    shutil.copy2(img_path, dest / img_path.name)

            stats[class_label]["total"] = n
            stats[class_label]["train"] = len(train_imgs)
            stats[class_label]["test"] = len(test_imgs)
            stats[class_label]["val"] = len(val_imgs)

            total_stats["total"] += n
            total_stats["train"] += len(train_imgs)
            total_stats["test"] += len(test_imgs)
            total_stats["val"] += len(val_imgs)

    print(f"\n{'Class':<30} {'Total':>7} {'Train':>7} {'Test':>7} {'Val':>7}")
    print("-" * 62)
    for label in sorted(stats.keys()):
        s = stats[label]
        print(
            f"{label:<30} {s['total']:>7} {s['train']:>7} {s['test']:>7} {s['val']:>7}"
        )
    print("-" * 62)
    print(
        f"{'TOTAL':<30} {total_stats['total']:>7} {total_stats['train']:>7} {total_stats['test']:>7} {total_stats['val']:>7}"
    )

    print(f"\nOutput directories:")
    print(f"   Train: {TRAIN_DIR}")
    print(f"   Test:  {TEST_DIR}")
    print(f"   Val:   {VAL_DIR}")
    print("\n Split complete.")


if __name__ == "__main__":
    main()

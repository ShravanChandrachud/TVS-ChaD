"""
CLIP Embedding Extraction
==========================
Loads all images from the organized dataset splits and extracts
512-dimensional feature vectors using the CLIP ViT-B/32 model
(via open-clip-torch). Embeddings are computed per split and
cached to disk so they only need to be extracted once.

Each image is preprocessed using CLIP's standard transforms
(resize, center crop, normalize) and passed through the image
encoder. Output vectors are L2-normalized.

Runs on GPU if available, falls back to CPU otherwise.
Run this ONCE before running any classifier script.

Input:  data/organized/<object>/<state>/(train|test|val)/*.jpg
Output: data/embeddings/clip_vit_b32/embeddings_(train|test|val).npy
        data/embeddings/clip_vit_b32/labels_(train|test|val).json
        data/embeddings/clip_vit_b32/paths_(train|test|val).json
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from utils import (
    ORGANIZED_DIR,
    save_split_embeddings,
    embeddings_exist,
)

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMBEDDING_MODEL_KEY = "clip_vit_b32"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OrganizedDataset(Dataset):
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, organized_dir: Path, split: str, transform=None):
        self.transform = transform
        self.samples = []

        for object_dir in sorted(organized_dir.iterdir()):
            if not object_dir.is_dir():
                continue
            for state_dir in sorted(object_dir.iterdir()):
                if not state_dir.is_dir():
                    continue
                split_dir = state_dir / split
                if not split_dir.exists():
                    continue
                label = f"{object_dir.name}/{state_dir.name}"
                for img_path in sorted(split_dir.iterdir()):
                    if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                        self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"No images found for split '{split}' in {organized_dir}")

        self.labels = sorted(set(s[1] for s in self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label, str(img_path)


def extract_from_split(model, preprocess, organized_dir: Path, split_name: str):
    print(f"\n {split_name.upper()} ")

    dataset = OrganizedDataset(organized_dir, split_name, transform=preprocess)
    print(f"   {len(dataset)} images across {len(dataset.labels)} classes")
    for label in dataset.labels:
        count = sum(1 for s in dataset.samples if s[1] == label)
        print(f"      {label}: {count}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if DEVICE == "cuda" else 0,
        pin_memory=DEVICE == "cuda",
    )

    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch_imgs, batch_labels, batch_paths in tqdm(
            loader, desc=f"   {split_name}"
        ):
            batch_imgs = batch_imgs.to(DEVICE)
            features = model.encode_image(batch_imgs)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()

            all_embeddings.append(features)
            all_labels.extend(batch_labels)
            all_paths.extend(batch_paths)

    embeddings = np.vstack(all_embeddings)
    print(f"   Extracted: {embeddings.shape[0]} × {embeddings.shape[1]}")

    save_split_embeddings(
        embeddings, all_labels, all_paths, split_name, EMBEDDING_MODEL_KEY
    )


def main():
    if embeddings_exist(EMBEDDING_MODEL_KEY):
        print("Embeddings already cached for all splits.")
        ans = input("Re-extract anyway? [y/N]: ").strip().lower()
        if ans != "y":
            return

    if not ORGANIZED_DIR.exists():
        print(f"ERROR: {ORGANIZED_DIR} not found.")
        print("Run organize_data.py first.")
        sys.exit(1)

    print("\nCLIP Embedding Extraction")
    print(f"Model:  {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    print(f"Device: {DEVICE}")
    print(f"Data:   {ORGANIZED_DIR}\n")

    import open_clip

    print("\nLoading CLIP model")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    for split_name in ["train", "test", "val"]:
        extract_from_split(model, preprocess, ORGANIZED_DIR, split_name)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\nAll embeddings extracted.")


if __name__ == "__main__":
    main()

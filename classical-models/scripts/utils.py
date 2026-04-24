"""
Shared Utilities for the Classification Pipeline
==================================================
Common functions and constants used across all classifier scripts
(extract_embeddings.py, classify_logreg.py, classify_svm.py,
classify_deep_cluster.py, video_inference.py).

Provides:
    - Project path constants (PROJECT_ROOT, ORGANIZED_DIR, EMBEDDINGS_DIR,
      OUTPUTS_DIR, MODELS_DIR)
    - Embedding I/O: save/load per-split embeddings as .npy + .json
    - Data preparation: load all splits, encode labels, scale features
      (scaler fit on train only to prevent data leakage), combine
      train+val for cross-validation
    - Evaluation: compute accuracy, F1, confusion matrix, classification
      report with explicit label handling for missing classes
    - Plotting: confusion matrix heatmaps, per-class F1 bar charts,
      t-SNE 2D visualizations with optional decision boundaries
    - Results saving: JSON serialization of metrics and hyperparameters
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ORGANIZED_DIR = PROJECT_ROOT / "data" / "organized"
TRAIN_DIR = ORGANIZED_DIR
TEST_DIR = ORGANIZED_DIR
VAL_DIR = ORGANIZED_DIR

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42


def get_embeddings_dir(model_name: str = "clip_vit_b32") -> Path:
    return EMBEDDINGS_DIR / model_name


def save_split_embeddings(
    embeddings: np.ndarray,
    labels: list,
    paths: list,
    split: str,
    model_name: str = "clip_vit_b32",
):
    cache_dir = get_embeddings_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.save(cache_dir / f"embeddings_{split}.npy", embeddings)
    with open(cache_dir / f"labels_{split}.json", "w") as f:
        json.dump(labels, f)
    with open(cache_dir / f"paths_{split}.json", "w") as f:
        json.dump(paths, f)

    print(f"   Saved {split}: {embeddings.shape[0]} samples → {cache_dir}")


def load_split_embeddings(
    split: str,
    model_name: str = "clip_vit_b32",
) -> tuple:
    cache_dir = get_embeddings_dir(model_name)
    emb_path = cache_dir / f"embeddings_{split}.npy"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"No cached {split} embeddings at {cache_dir}\n"
            f"Run extract_embeddings.py first."
        )

    embeddings = np.load(emb_path)
    with open(cache_dir / f"labels_{split}.json") as f:
        labels = json.load(f)
    with open(cache_dir / f"paths_{split}.json") as f:
        paths = json.load(f)

    print(
        f"Loaded {split} embeddings: {embeddings.shape[0]} samples, dim {embeddings.shape[1]}"
    )
    return embeddings, labels, paths


def embeddings_exist(model_name: str = "clip_vit_b32") -> bool:
    cache_dir = get_embeddings_dir(model_name)
    return all(
        (cache_dir / f"embeddings_{split}.npy").exists()
        for split in ("train", "test", "val")
    )



def prepare_data(model_name: str = "clip_vit_b32") -> dict:
    X_train, y_train_raw, paths_train = load_split_embeddings("train", model_name)
    X_test, y_test_raw, paths_test = load_split_embeddings("test", model_name)
    X_val, y_val_raw, paths_val = load_split_embeddings("val", model_name)

    le = LabelEncoder()
    le.fit(y_train_raw) 
    label_names = le.classes_.tolist()

    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    y_val = le.transform(y_val_raw)

    print(f"\nClasses ({len(label_names)}):")
    for i, name in enumerate(label_names):
        n_tr = (y_train == i).sum()
        n_te = (y_test == i).sum()
        n_va = (y_val == i).sum()
        print(f"   [{i:2d}] {name}: {n_tr} train / {n_te} test / {n_va} val")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])

    print(
        f"\nSplit sizes: {X_train_scaled.shape[0]} train / {X_test_scaled.shape[0]} test / {X_val_scaled.shape[0]} val"
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_val": X_val_scaled,
        "X_trainval": X_trainval,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "y_trainval": y_trainval,
        "label_encoder": le,
        "label_names": label_names,
        "scaler": scaler,
    }

def evaluate_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    label_names: list,
    clf_name: str,
    split_name: str = "test",
) -> dict:
    y_pred = clf.predict(X)

    all_labels = list(range(len(label_names)))

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro", labels=all_labels, zero_division=0)
    f1_weighted = f1_score(
        y, y_pred, average="weighted", labels=all_labels, zero_division=0
    )
    report = classification_report(
        y,
        y_pred,
        target_names=label_names,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y, y_pred, target_names=label_names, labels=all_labels, zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=all_labels)

    print(f"\n{'─' * 50}")
    print(f"Results: {clf_name} [{split_name}]")
    print(f"{'─' * 50}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"F1 (macro):      {f1_macro:.4f}")
    print(f"F1 (weighted):   {f1_weighted:.4f}")
    print(f"\n{report_str}")

    return {
        "classifier": clf_name,
        "split": split_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved: {save_path}")


def plot_per_class_f1(results_dict, label_names, save_path):
    report = results_dict["classification_report"]
    f1_scores = [report.get(label, {}).get("f1-score", 0) for label in label_names]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(label_names))
    bars = ax.bar(x, f1_scores, color="steelblue", alpha=0.85)

    for bar, score in zip(bars, f1_scores):
        if score < 0.5:
            bar.set_color("indianred")
        elif score < 0.7:
            bar.set_color("orange")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(
        f"Per-Class F1 — {results_dict['classifier']}\n"
        f"Accuracy: {results_dict['accuracy']:.3f} | F1 (macro): {results_dict['f1_macro']:.3f}",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved: {save_path}")


def plot_tsne(X, y_true, y_pred, label_names, title, save_path, clf_2d=None, X_2d=None):
    from matplotlib.colors import ListedColormap

    colors = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#42d4f4",
        "#f032e6",
        "#bfef45",
        "#fabed4",
        "#469990",
        "#dcbeff",
        "#9A6324",
    ]
    n_classes = len(label_names)
    cmap = ListedColormap(colors[:n_classes])

    if X_2d is None:
        print("   Running t-SNE")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 9))

    if clf_2d is not None:
        h = 0.5
        x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
        y_min, y_max = X_2d[:, 1].min() - 2, X_2d[:, 1].max() + 2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(
            xx, yy, Z, alpha=0.08, cmap=cmap, levels=np.arange(-0.5, n_classes + 0.5, 1)
        )
        ax.contour(xx, yy, Z, colors="gray", linewidths=0.3, alpha=0.5)

    ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_pred,
        cmap=cmap,
        s=15,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.2,
    )

    handles = [
        plt.scatter([], [], c=colors[i], s=40, edgecolors="k", linewidths=0.3)
        for i in range(n_classes)
    ]
    ax.legend(
        handles, label_names, fontsize=7, loc="best", markerscale=1.2, frameon=True
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {save_path}")

    return X_2d


def save_results(results_dict, save_dir: Path, prefix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"{prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"   Results saved: {json_path}")

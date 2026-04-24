"""
Video Inference — Classify Every Frame with All Saved Models
==============================================================
Takes a cooking video from root/videos/, extracts frames at a configurable
FPS rate (default 2 FPS), computes CLIP embeddings for each frame, then
runs every saved model (LogReg, SVM Linear, SVM RBF, DEC) on all frames.

For each model, builds a "memory bank" that tracks object state changes
over time. State is tracked per-object independently — each object
(eggs, pan, bowl, etc.) maintains its own state history, and a transition
is only logged when that specific object's predicted state differs from
its own previous state (not from whatever object appeared in the prior frame).

Each transition includes a confidence score: true softmax probabilities
for LogReg, decision-function-derived pseudo-confidence for SVM, and
inverse centroid distance for DEC.

Outputs a formatted .txt memory bank per model (matching the project's
memory bank table format) plus a comprehensive JSON with per-frame
predictions from all models.

Input:  root/videos/*.mp4 (first video found)
        root/models/*.pkl and *.pt (all saved models)
Output: outputs/video_inference_<timestamp>/
            <model>_memory_bank.txt
            <model>_memory_bank.json
            all_frames_predictions.json
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from PIL import Image

from utils import (
    PROJECT_ROOT,
    OUTPUTS_DIR,
    MODELS_DIR,
)

VIDEOS_DIR = PROJECT_ROOT / "videos"
FPS_EXTRACT = 2              
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
COSINE_THRESHOLD = 0.90       

DEC_HIDDEN_1 = 1024
DEC_HIDDEN_2 = 512
DEC_LATENT_DIM = 128


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_video() -> Path:
    """Find the first video file in videos/ folder."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Videos directory not found: {VIDEOS_DIR}")

    for f in sorted(VIDEOS_DIR.iterdir()):
        if f.suffix.lower() in video_extensions:
            return f

    raise FileNotFoundError(f"No video files found in {VIDEOS_DIR}")


def extract_frames(video_path: Path, target_fps: int) -> list:
    """
    Extract frames from video at target FPS.
    Returns list of (frame_number, timestamp_seconds, PIL_image).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps

    frame_interval = int(original_fps / target_fps)

    print(f"   Video: {video_path.name}")
    print(f"   Original FPS: {original_fps:.1f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Extracting at {target_fps} FPS (every {frame_interval} frames)")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / original_fps
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append((frame_idx, timestamp, pil_img))

        frame_idx += 1

    cap.release()
    print(f"   Extracted {len(frames)} frames")
    return frames


def extract_clip_embeddings(frames: list, model, preprocess) -> np.ndarray:
    """Extract CLIP embeddings for all frames. Returns (N, 512) array."""
    print(f"\n   Extracting CLIP embeddings on {DEVICE}")
    embeddings = []

    batch_size = 32
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_imgs = torch.stack([preprocess(f[2]) for f in batch_frames]).to(DEVICE)

        with torch.no_grad():
            features = model.encode_image(batch_imgs)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            print(f"      Processed {min(i + batch_size, len(frames))}/{len(frames)} frames")

    embeddings = np.vstack(embeddings)
    print(f"   Done: {embeddings.shape[0]} embeddings × {embeddings.shape[1]} dims")
    return embeddings


def format_timestamp(seconds: float) -> str:
    """Convert seconds to M:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def detect_change_frames(embeddings: np.ndarray, threshold: float = COSINE_THRESHOLD) -> list:
    """
    Compare consecutive frame embeddings using cosine similarity.
    Returns list of indices where similarity drops below threshold,
    indicating a potential state change. Frame 0 is always included.
    """
    change_indices = [0]  # First frame is always a change (initial state)

    for i in range(1, len(embeddings)):
        dot = np.dot(embeddings[i], embeddings[i - 1])
        norm_a = np.linalg.norm(embeddings[i])
        norm_b = np.linalg.norm(embeddings[i - 1])

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = dot / (norm_a * norm_b)

        if similarity < threshold:
            change_indices.append(i)

    return change_indices


def build_memory_bank(
    frame_data: list,
    predictions: list,
    confidences: list,
    label_names: list,
    change_indices: list,
) -> list:
    """
    Build memory bank JSON from frame predictions.
    Only processes frames flagged as state changes by cosine similarity.
    Tracks state per object independently.

    Returns list of state change entries.
    """
    memory_bank = []
    object_states = {}  # Track previous state PER OBJECT

    for i in change_indices:
        frame_num, timestamp, _ = frame_data[i]
        current_label = label_names[predictions[i]]
        confidence = float(confidences[i])

        parts = current_label.split("/")
        obj = parts[0] if len(parts) > 0 else "unknown"
        state = parts[1] if len(parts) > 1 else parts[0]

        prev_state = object_states.get(obj, None)

        if state != prev_state:
            entry = {
                "time": format_timestamp(timestamp),
                "object": obj,
                "new_state": state,
                "prev_state": prev_state if prev_state else "none",
                "frame": frame_num,
                "confidence": round(confidence, 3),
                "timestamp_seconds": round(timestamp, 2),
            }
            memory_bank.append(entry)
            object_states[obj] = state

    return memory_bank


def load_sklearn_model(model_path: Path) -> dict:
    """Load a scikit-learn model saved with joblib."""
    return joblib.load(model_path)


def predict_sklearn(model_data: dict, embeddings: np.ndarray) -> tuple:
    """Run prediction with a sklearn model (LogReg or SVM).
    Returns (predictions, confidences) where confidences are 0-1 scores."""
    clf = model_data["classifier"]
    scaler = model_data["scaler"]
    X_scaled = scaler.transform(embeddings)
    predictions = clf.predict(X_scaled)

    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_scaled)
        confidences = probas.max(axis=1)
    elif hasattr(clf, "decision_function"):
        decision = clf.decision_function(X_scaled)
        if decision.ndim == 1:
            confidences = 1 / (1 + np.exp(-np.abs(decision)))
        else:
            exp_d = np.exp(decision - decision.max(axis=1, keepdims=True))
            softmax = exp_d / exp_d.sum(axis=1, keepdims=True)
            confidences = softmax.max(axis=1)
    else:
        confidences = np.ones(len(predictions))

    return predictions, confidences


def load_dec_model(model_path: Path) -> dict:
    """Load DEC model."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    encoder = nn.Sequential(
        nn.Linear(512, DEC_HIDDEN_1), nn.BatchNorm1d(DEC_HIDDEN_1), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(DEC_HIDDEN_1, DEC_HIDDEN_2), nn.BatchNorm1d(DEC_HIDDEN_2), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(DEC_HIDDEN_2, DEC_LATENT_DIM),
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(DEVICE)
    encoder.eval()

    raw_mapping = checkpoint["mapping"]
    mapping = {int(k): int(v) for k, v in raw_mapping.items()}

    return {
        "encoder": encoder,
        "cluster_centers": checkpoint["cluster_centers"],
        "mapping": mapping,
    }

def predict_dec(dec_data: dict, embeddings: np.ndarray, label_names: list):
    """Run prediction with DEC model. Returns (predictions, confidences)."""
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    encoder = dec_data["encoder"]
    encoder.eval()
    with torch.no_grad():
        z = encoder(torch.FloatTensor(X_scaled).to(DEVICE)).cpu().numpy()

    cluster_centers = dec_data["cluster_centers"]
    dists = cdist(z, cluster_centers)
    cluster_ids = dists.argmin(axis=1)

    inv_dists = 1.0 / (1.0 + dists)
    confidences_raw = inv_dists / inv_dists.sum(axis=1, keepdims=True)
    confidences = confidences_raw.max(axis=1)

    mapping = dec_data["mapping"]
    mapped = np.array([mapping.get(c, 0) for c in cluster_ids])

    return mapped, confidences


def main():
    print("=" * 60)
    print("Video Inference — All Models")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    video_path = find_video()

    print("\nExtracting frames")
    frames = extract_frames(video_path, FPS_EXTRACT)

    print("\nLoading CLIP model")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()

    embeddings = extract_clip_embeddings(frames, clip_model, preprocess)

    del clip_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\nDetecting state changes (cosine threshold = {COSINE_THRESHOLD})")
    change_indices = detect_change_frames(embeddings, COSINE_THRESHOLD)
    print(f"   Frames with potential state change: {len(change_indices)} / {len(frames)}")
    print(f"   Reduction: {100 * (1 - len(change_indices) / len(frames)):.1f}% of frames skipped")

    print("\nDiscovering saved models")
    models = {}

    for pkl_path in sorted(MODELS_DIR.glob("*.pkl")):
        name = pkl_path.stem.replace("_model", "")
        print(f"   Found: {pkl_path.name} → {name}")
        models[name] = {"type": "sklearn", "path": pkl_path}

    for pt_path in sorted(MODELS_DIR.glob("*.pt")):
        name = pt_path.stem.replace("_model", "")
        print(f"   Found: {pt_path.name} → {name}")
        models[name] = {"type": "dec", "path": pt_path}

    if not models:
        print("    No models found in models/ folder!")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUTS_DIR / f"video_inference_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {}

    for model_name, model_info in models.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {model_name}")
        print(f"{'=' * 60}")

        if model_info["type"] == "sklearn":
            model_data = load_sklearn_model(model_info["path"])
            label_names = model_data["label_names"]
            predictions, confidences = predict_sklearn(model_data, embeddings)

        elif model_info["type"] == "dec":
            dec_data = load_dec_model(model_info["path"])
            sklearn_model = None
            for m in models.values():
                if m["type"] == "sklearn":
                    sklearn_model = load_sklearn_model(m["path"])
                    break
            if sklearn_model:
                label_names = sklearn_model["label_names"]
            else:
                label_names = [f"class_{i}" for i in range(13)]

            predictions, confidences = predict_dec(dec_data, embeddings, label_names)

        memory_bank = build_memory_bank(frames, predictions, confidences, label_names, change_indices)

        print(f"\n   State changes detected: {len(memory_bank)}")

        txt_path = run_dir / f"{model_name}_memory_bank.txt"
        with open(txt_path, "w", encoding="utf-8") as txt:
            txt.write(f"MEMORY BANK - {video_path.stem}\n")
            txt.write(f"Model: {model_name}\n")
            txt.write(f"Generated: {datetime.now().isoformat()}\n")

            objects_seen = set(e["object"] for e in memory_bank)
            txt.write(f"Instances tracked: {len(objects_seen)}\n")
            txt.write(f"Total transitions: {len(memory_bank)}\n")

            line = "-" * 105
            txt.write(f"{line}\n")
            txt.write(f"{'Time':<9}| {'Object':<13}| {'New State':<22}| {'Prev State':<22}| {'Frame':<9}| {'Confidence'}\n")
            txt.write(f"{line}\n")

            for entry in memory_bank:
                conf_str = f"{entry['confidence']:.1%}"
                txt.write(
                    f"{entry['time']:<9}| "
                    f"{entry['object']:<13}| "
                    f"{entry['new_state']:<22}| "
                    f"{entry['prev_state']:<22}| "
                    f"{entry['frame']:<9}| "
                    f"{conf_str}\n"
                )

            txt.write(f"{line}\n")

        print(f"   Saved: {txt_path}")

        # Save memory bank
        bank_path = run_dir / f"{model_name}_memory_bank.json"
        with open(bank_path, "w") as f:
            json.dump({
                "model": model_name,
                "video": video_path.name,
                "fps_extract": FPS_EXTRACT,
                "cosine_threshold": COSINE_THRESHOLD,
                "total_frames_extracted": len(frames),
                "change_frames_detected": len(change_indices),
                "state_changes": len(memory_bank),
                "memory_bank": memory_bank,
            }, f, indent=2)
        print(f"   Saved: {bank_path}")

        # Store full per-frame predictions
        all_predictions[model_name] = {
            "label_names": label_names,
            "per_frame": [
                {
                    "frame": frames[i][0],
                    "time": format_timestamp(frames[i][1]),
                    "timestamp_seconds": round(frames[i][1], 2),
                    "predicted_class": label_names[predictions[i]],
                    "predicted_index": int(predictions[i]),
                    "confidence": round(float(confidences[i]), 4),
                }
                for i in range(len(frames))
            ],
        }

    # Save all predictions
    all_path = run_dir / "all_frames_predictions.json"
    with open(all_path, "w") as f:
        json.dump({
            "video": video_path.name,
            "fps_extract": FPS_EXTRACT,
            "total_frames": len(frames),
            "models": list(models.keys()),
            "predictions": all_predictions,
        }, f, indent=2)
    print(f"\nAll predictions saved: {all_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Video: {video_path.name}")
    print(f"Frames extracted: {len(frames)} (at {FPS_EXTRACT} FPS)")
    print(f"Cosine threshold: {COSINE_THRESHOLD}")
    print(f"Change frames detected: {len(change_indices)} / {len(frames)} ({100 * len(change_indices) / len(frames):.1f}%)")
    print(f"Models run: {len(models)}")
    for model_name in models:
        bank = json.loads((run_dir / f"{model_name}_memory_bank.json").read_text())
        print(f"   {model_name}: {bank['state_changes']} state changes")
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n{'!' * 60}")
        print(f"ERROR: {e}")
        print(f"{'!' * 60}")
        traceback.print_exc()
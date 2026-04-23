"""
Video State Tracker — Main Entry Point

Usage:
    python main.py --video input/test_h264.mp4 --task "making an omelette from scratch"

With a pre-generated schema:
    python main.py --video input/test_h264.mp4 --schema output/schema.json

Override defaults:
    python main.py --video input/test_h264.mp4 --task "making an omelette" --fps 3 --threshold 0.80
"""

import argparse
import json
from pathlib import Path

import config
from backends import get_embedding_backend, get_llm_backend, get_vlm_backend
from schema_generator import generate_schema
from core.state_tracker import StateTracker
from utils.video import extract_frames


def main():
    parser = argparse.ArgumentParser(description="Video State Tracker")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--task", type=str, default=None,
                        help="Task description (e.g., 'making an omelette')")
    parser.add_argument("--schema", type=str, default=None,
                        help="Path to a pre-generated schema JSON (skips LLM generation)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for memory bank JSON")
    parser.add_argument("--fps", type=float, default=None,
                        help=f"Sample FPS (default: {config.SAMPLE_FPS})")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Similarity threshold (default: {config.SIMILARITY_THRESHOLD})")
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    if args.task is None and args.schema is None:
        print("Error: Provide either --task (for LLM schema generation) or --schema (pre-made).")
        return

    # ── Step 1: Get the schema ───────────────────────────────
    if args.schema:
        print(f"[Main] Loading schema from {args.schema}")
        with open(args.schema, "r") as f:
            schema = json.load(f)
    else:
        print(f"[Main] Generating schema for: '{args.task}'")
        llm = get_llm_backend(config.LLM_BACKEND)
        schema = generate_schema(llm, args.task)

    # Save schema for reproducibility
    schema_out = config.OUTPUT_DIR / "schema.json"
    schema_out.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_out, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[Main] Schema saved to {schema_out}")

    # ── Step 2: Extract frames ───────────────────────────────
    sample_fps = args.fps or config.SAMPLE_FPS
    frames = extract_frames(video_path, sample_fps=sample_fps)

    # ── Step 3: Load all backends (both stay in VRAM) ────────
    print(f"\n[Main] Loading embedding backend: {config.EMBEDDING_BACKEND}")
    embedder = get_embedding_backend(config.EMBEDDING_BACKEND)
    embedder.load()

    print(f"[Main] Loading VLM backend: {config.VLM_BACKEND}")
    vlm = get_vlm_backend(config.VLM_BACKEND)

    # ── Step 4: Run the tracker ──────────────────────────────
    threshold = args.threshold or config.SIMILARITY_THRESHOLD
    tracker = StateTracker(
        embedding_backend=embedder,
        vlm_backend=vlm,
        schema=schema,
        threshold=threshold,
    )

    print(f"\n{'='*60}")
    print(f"  TRACKING START")
    print(f"  Video: {video_path.name}")
    print(f"  Frames: {len(frames)} @ {sample_fps} FPS")
    print(f"  Threshold: {threshold}")
    print(f"  Embedding: {config.EMBEDDING_BACKEND}")
    print(f"  VLM: {config.VLM_BACKEND}")
    print(f"  Objects: {', '.join(schema['objects'].keys())}")
    print(f"{'='*60}\n")

    tracker.process_frames(frames)

    # ── Step 5: Save output ──────────────────────────────────
    memory_bank = tracker.get_memory_bank()
    output_path = Path(args.output) if args.output else config.OUTPUT_DIR / "memory_bank.json"
    memory_bank.save(output_path)
    memory_bank.save_table(output_path.with_suffix(".txt"))

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Memory bank: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

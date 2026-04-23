"""
Video utilities — frame extraction at a target FPS.
"""

import cv2
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

import config


@dataclass
class Frame:
    """A single extracted frame with metadata."""
    image: Image.Image
    index: int          # Sequential index in the sampled sequence
    frame_number: int   # Original frame number in the video
    timestamp: float    # Seconds into the video


def extract_frames(video_path: str | Path, sample_fps: float = None) -> list[Frame]:
    """
    Extract frames from a video at a given sample rate.
    
    Args:
        video_path: Path to the video file.
        sample_fps: Frames per second to sample. Defaults to config.SAMPLE_FPS.
    
    Returns:
        List of Frame objects.
    """
    sample_fps = sample_fps or config.SAMPLE_FPS
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps

    # How many native frames to skip between samples
    frame_interval = max(1, int(native_fps / sample_fps))

    frames = []
    frame_number = 0
    sample_index = 0

    print(f"[Video] {video_path.name}: {native_fps:.1f} FPS, "
          f"{total_frames} frames, {duration:.1f}s duration")
    print(f"[Video] Sampling at {sample_fps} FPS (every {frame_interval} frames)")

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            # Convert BGR (OpenCV) → RGB (PIL)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            timestamp = frame_number / native_fps

            frames.append(Frame(
                image=pil_image,
                index=sample_index,
                frame_number=frame_number,
                timestamp=round(timestamp, 3),
            ))
            sample_index += 1

        frame_number += 1

    cap.release()
    print(f"[Video] Extracted {len(frames)} frames.")
    return frames

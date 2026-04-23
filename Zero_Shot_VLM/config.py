"""
Global configuration for the Video State Tracker pipeline.
All tunable parameters live here — nothing is hardcoded in modules.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ──────────────────────────────────────────────
# Video Sampling
# ──────────────────────────────────────────────
SAMPLE_FPS = 2  # Frames per second to sample from the video

# ──────────────────────────────────────────────
# Change Detection
# ──────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85  # Below this = "something changed", trigger VLM
EMBEDDING_DEVICE = "cuda"    # "cuda" or "cpu"

# ──────────────────────────────────────────────
# Backend Selection (swap these to change models)
# ──────────────────────────────────────────────
EMBEDDING_BACKEND = "dinov2"      # Options: "dinov2", "vjepa2", "clip"
LLM_BACKEND = "gemini"            # Options: "gemini", "groq", "openai"
VLM_BACKEND = "auto"              # Options: "auto" (Gemini→Gemma4 fallback), "gemini", "gemma4"

# ──────────────────────────────────────────────
# Model-Specific Config
# ──────────────────────────────────────────────
DINOV2_MODEL = "facebook/dinov2-small"

# Gemini API (gemini-2.0-flash is DEPRECATED as of March 2026)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = ""  # Set via env var GEMINI_API_KEY or paste here

# Gemma 4 Local VLM
GEMMA4_MODEL = "google/gemma-4-e4b-it"  # ~3.5GB VRAM, fits on 8GB GPUs
GEMMA4_DEVICE = "cuda"
GEMMA4_DTYPE = "bfloat16"                # "bfloat16" or "float16"

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY = ""    # Set via env var GROQ_API_KEY or paste here

# ──────────────────────────────────────────────
# VLM Verification
# ──────────────────────────────────────────────
VLM_MAX_RETRIES = 2       # Retry VLM call if JSON parsing fails
VLM_TEMPERATURE = 0.1     # Low temp for deterministic state assignment

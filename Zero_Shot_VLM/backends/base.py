"""
Abstract base classes for all backends.
Every backend implements one of these interfaces — that's the plug-and-play contract.
"""

from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


class EmbeddingBackend(ABC):
    """Takes an image, returns a fixed-size embedding vector."""

    @abstractmethod
    def load(self):
        """Load model into memory. Called once at pipeline start."""
        ...

    @abstractmethod
    def embed(self, image: Image.Image) -> np.ndarray:
        """Compute embedding for a single PIL image. Returns 1-D numpy array."""
        ...

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings. Default implementation provided."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))

    @abstractmethod
    def unload(self):
        """Free GPU memory. Called when switching to VLM phase."""
        ...


class LLMBackend(ABC):
    """Text-in, text-out. Used for schema generation."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Send a text prompt, get a text response."""
        ...


class VLMBackend(ABC):
    """Image + text in, text out. Used for state verification."""

    @abstractmethod
    def query(self, images: list[Image.Image], prompt: str, system_prompt: str = "") -> str:
        """
        Send one or more images with a text prompt, get a text response.
        images[0] = anchor frame, images[1] = current frame (for two-frame comparison).
        """
        ...

"""
Change Detector — compares current frame embedding against the anchor frame.
Flags frames where similarity drops below threshold (something visually changed).
"""

import numpy as np
from PIL import Image

from backends.base import EmbeddingBackend
import config


class ChangeDetector:
    """
    Manages the rolling anchor and detects visual changes via embedding comparison.
    """

    def __init__(self, embedding_backend: EmbeddingBackend, threshold: float = None):
        self.embedder = embedding_backend
        self.threshold = threshold or config.SIMILARITY_THRESHOLD
        self.anchor_embedding: np.ndarray | None = None

    def set_anchor(self, image: Image.Image):
        """Set (or update) the anchor frame. Called at start and after confirmed transitions."""
        self.anchor_embedding = self.embedder.embed(image)

    def check_change(self, image: Image.Image) -> tuple[bool, float]:
        """
        Compare a frame against the anchor.
        
        Returns:
            (changed: bool, similarity: float)
            changed=True if similarity < threshold (something visually different).
        """
        if self.anchor_embedding is None:
            raise RuntimeError("Anchor not set. Call set_anchor() first.")

        current_embedding = self.embedder.embed(image)
        sim = self.embedder.similarity(self.anchor_embedding, current_embedding)

        changed = sim < self.threshold
        return changed, sim

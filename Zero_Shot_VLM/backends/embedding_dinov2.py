"""
DINOv2-small embedding backend.
Uses the CLS token as the frame embedding.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from backends.base import EmbeddingBackend
import config


class DINOv2Backend(EmbeddingBackend):

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = config.EMBEDDING_DEVICE

    def load(self):
        print(f"[DINOv2] Loading {config.DINOV2_MODEL} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(config.DINOV2_MODEL)
        self.model = AutoModel.from_pretrained(config.DINOV2_MODEL).to(self.device)
        self.model.eval()
        print("[DINOv2] Ready.")

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().cpu().numpy()

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            print("[DINOv2] Unloaded.")

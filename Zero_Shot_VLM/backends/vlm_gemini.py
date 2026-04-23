"""
Gemini VLM backend — used for two-frame state verification.
Sends images + prompt via the google-genai SDK.
Raises on failure so the fallback wrapper can catch and switch.
"""

import os
import google.generativeai as genai
from PIL import Image

from backends.base import VLMBackend
import config


class GeminiVLMBackend(VLMBackend):

    def __init__(self):
        api_key = config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in config or environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        print(f"[Gemini VLM] Initialized with {config.GEMINI_MODEL}")

    def query(self, images: list[Image.Image], prompt: str, system_prompt: str = "") -> str:
        content_parts = []

        if system_prompt:
            content_parts.append(system_prompt + "\n\n")

        for img in images:
            content_parts.append(img)

        content_parts.append(prompt)

        response = self.model.generate_content(
            content_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=config.VLM_TEMPERATURE,
            ),
        )
        return response.text

"""
Gemini LLM backend — used for schema generation.
Uses the google-genai SDK (lightweight REST wrapper).
"""

import os
import json
import google.generativeai as genai

from backends.base import LLMBackend
import config


class GeminiLLMBackend(LLMBackend):

    def __init__(self):
        api_key = config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in config or environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
            ),
        )
        return response.text

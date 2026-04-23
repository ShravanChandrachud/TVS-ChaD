"""
Fallback VLM — tries Gemini API first, falls back to Gemma 4 local.

Handles three failure modes:
1. No API key set → skip Gemini entirely, load Gemma 4 immediately.
2. API error during a call (rate limit, network, etc.) → switch to Gemma 4 for
   the rest of the session (don't keep hammering a broken API).
3. Gemma 4 init failure → raise, nothing to fall back to.
"""

from PIL import Image
from backends.base import VLMBackend


class FallbackVLMBackend(VLMBackend):

    def __init__(self):
        self.primary: VLMBackend | None = None
        self.fallback: VLMBackend | None = None
        self.using_fallback = False

        # Try to init Gemini (API)
        try:
            from backends.vlm_gemini import GeminiVLMBackend
            self.primary = GeminiVLMBackend()
            print("[FallbackVLM] Primary: Gemini API")
        except Exception as e:
            print(f"[FallbackVLM] Gemini unavailable ({e}). Will use Gemma 4 local.")
            self._activate_fallback()

    def _activate_fallback(self):
        """Load Gemma 4 local model. Only called once."""
        if self.fallback is not None:
            return  # Already loaded

        print("[FallbackVLM] Activating fallback: Gemma 4 local...")
        from backends.vlm_gemma4 import Gemma4Backend
        self.fallback = Gemma4Backend()
        self.using_fallback = True
        print("[FallbackVLM] Fallback active.")

    def query(self, images: list[Image.Image], prompt: str, system_prompt: str = "") -> str:
        # If already on fallback, go straight to Gemma 4
        if self.using_fallback:
            return self.fallback.query(images, prompt, system_prompt)

        # Try Gemini first
        try:
            return self.primary.query(images, prompt, system_prompt)
        except Exception as e:
            print(f"\n[FallbackVLM] Gemini call failed: {e}")
            print("[FallbackVLM] Switching to Gemma 4 local for remaining calls.")
            self._activate_fallback()
            return self.fallback.query(images, prompt, system_prompt)

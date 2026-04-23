"""
Backend factory — instantiate the right backend from a config string.
"""

from backends.base import EmbeddingBackend, LLMBackend, VLMBackend


def get_embedding_backend(name: str) -> EmbeddingBackend:
    if name == "dinov2":
        from backends.embedding_dinov2 import DINOv2Backend
        return DINOv2Backend()
    # Add more here:
    # elif name == "vjepa2":
    #     from backends.embedding_vjepa2 import VJEPA2Backend
    #     return VJEPA2Backend()
    raise ValueError(f"Unknown embedding backend: {name}")


def get_llm_backend(name: str) -> LLMBackend:
    if name == "gemini":
        from backends.llm_gemini import GeminiLLMBackend
        return GeminiLLMBackend()
    # elif name == "groq":
    #     from backends.llm_groq import GroqLLMBackend
    #     return GroqLLMBackend()
    raise ValueError(f"Unknown LLM backend: {name}")


def get_vlm_backend(name: str) -> VLMBackend:
    if name == "auto":
        from backends.vlm_fallback import FallbackVLMBackend
        return FallbackVLMBackend()
    elif name == "gemini":
        from backends.vlm_gemini import GeminiVLMBackend
        return GeminiVLMBackend()
    elif name == "gemma4":
        from backends.vlm_gemma4 import Gemma4Backend
        return Gemma4Backend()
    # elif name == "qwen":
    #     from backends.vlm_qwen import QwenVLMBackend
    #     return QwenVLMBackend()
    raise ValueError(f"Unknown VLM backend: {name}")

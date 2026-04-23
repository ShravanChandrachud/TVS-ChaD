"""
Gemma 4 E4B local VLM backend.
Runs entirely on-device — no API calls, no rate limits.
Uses ~3.5GB VRAM, coexists with DINOv2-small (~0.1GB) on 8GB GPUs.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from backends.base import VLMBackend
import config


class Gemma4Backend(VLMBackend):

    def __init__(self):
        self.device = config.GEMMA4_DEVICE
        self.dtype = getattr(torch, config.GEMMA4_DTYPE)

        print(f"[Gemma4] Loading {config.GEMMA4_MODEL} on {self.device} ({config.GEMMA4_DTYPE})...")
        self.processor = AutoProcessor.from_pretrained(config.GEMMA4_MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.GEMMA4_MODEL,
            dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()
        print("[Gemma4] Ready.")

    def query(self, images: list[Image.Image], prompt: str, system_prompt: str = "") -> str:
        user_content = []

        for i, img in enumerate(images):
            label = "ANCHOR FRAME" if i == 0 else "CURRENT FRAME"
            user_content.append({"type": "text", "text": f"[{label}]"})
            user_content.append({"type": "image", "image": img})

        user_content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": user_content})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=config.VLM_TEMPERATURE,
                do_sample=True,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()

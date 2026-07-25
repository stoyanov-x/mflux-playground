"""mflux inference wrapper.

This module is the one piece that changes when you swap the inference
backend. The rest of the service (routers, job queue) talks only to
the `InferenceEngine` interface defined here.

To port to a cloud target (Replicate / Fal.ai / RunPod) you'd:

  1. Create a new class implementing the same `generate()` signature.
  2. Swap it in via `main.py` or config.
  3. The FastAPI routers and job store stay untouched.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Optional

import structlog
from PIL import Image

from app.config import settings

logger = structlog.get_logger(__name__)


class InferenceEngine:
    """Wraps an mflux model for local inference on Apple Silicon."""

    def __init__(self) -> None:
        self._model = None
        self._model_alias = settings.model_alias
        self._quantize = settings.quantize
        self._default_lora_paths = [
            p.strip() for p in settings.lora_paths.split(",") if p.strip()
        ]
        logger.info(
            "inference_init",
            model_alias=self._model_alias,
            quantize=self._quantize,
            default_loras=self._default_lora_paths,
        )

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Load (or reload) the Flux model with configured options."""
        if self._model is not None:
            return  # already loaded; could add reload-on-config-change

        from mflux import Flux1  # slow import – only when needed

        kwargs = {"quantize": self._quantize} if self._quantize else {}
        self._model = Flux1.from_alias(self._model_alias, **kwargs)
        logger.info("model_loaded", alias=self._model_alias, quantize=self._quantize)

    def unload_model(self) -> None:
        self._model = None
        logger.info("model_unloaded")

    # ── Core generation ──────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        lora_paths: Optional[list[str]] = None,
        lora_scales: Optional[list[float]] = None,
        reference_image_b64: Optional[str] = None,
        denoise_strength: float = 0.75,
        num_inference_steps: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 3.5,
    ) -> tuple[Image.Image, int, float]:
        """
        Run a single generation.

        Returns
        -------
        (image, seed_used, elapsed_seconds)
        """
        from mflux import Config

        # Merge default LoRAs with any request-specific ones.
        all_lora_paths = list(self._default_lora_paths)
        if lora_paths:
            all_lora_paths.extend(lora_paths)

        all_lora_scales = lora_scales or [0.8] * len(all_lora_paths)

        steps = num_inference_steps or settings.default_num_steps
        h = height or settings.default_height
        w = width or settings.default_width
        seed = seed or int(time.time() * 1000) % (2**31)

        cfg = Config(
            num_inference_steps=steps,
            height=h,
            width=w,
            guidance_scale=guidance_scale,
        )

        start = time.perf_counter()

        if reference_image_b64 is not None:
            # Image-to-image branch
            raw = base64.b64decode(reference_image_b64)
            ref_image = Image.open(io.BytesIO(raw)).convert("RGB")
            image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                config=cfg,
                image=ref_image,
                strength=denoise_strength,
            )
        else:
            # Text-to-image branch (with optional LoRAs)
            image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                config=cfg,
                lora_paths=all_lora_paths or None,
                lora_scales=all_lora_scales or None,
            )

        elapsed = time.perf_counter() - start

        logger.info(
            "generation_complete",
            prompt_trunc=prompt[:60],
            seed=seed,
            elapsed=f"{elapsed:.2f}s",
            loras=all_lora_paths,
        )

        return image, seed, elapsed

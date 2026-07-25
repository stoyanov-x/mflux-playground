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
import tempfile
import time
from pathlib import Path
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
            return  # already loaded

        # Current mflux API (mid-2026): Flux1 takes ModelConfig, not from_alias()
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1

        model_config = ModelConfig.from_name(
            model_name=self._model_alias,
            base_model=None,
        )

        self._model = Flux1(
            model_config=model_config,
            quantize=self._quantize,
            model_path=None,
            lora_paths=self._default_lora_paths or None,
            lora_scales=[0.8] * len(self._default_lora_paths) or None,
        )
        logger.info(
            "model_loaded",
            alias=self._model_alias,
            quantize=self._quantize,
            loras=self._default_lora_paths,
        )

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
        denoise_strength: float = 0.4,
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
        steps = num_inference_steps or settings.default_num_steps
        h = height or settings.default_height
        w = width or settings.default_width
        seed = seed or int(time.time() * 1000) % (2**31)

        # ── img2img: save base64 ref to temp file ────────────
        image_path: Path | None = None
        temp_file = None
        if reference_image_b64 is not None:
            raw = base64.b64decode(reference_image_b64)
            ref = Image.open(io.BytesIO(raw)).convert("RGB")
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            ref.save(temp_file, format="PNG")
            image_path = Path(temp_file.name)

        start = time.perf_counter()

        try:
            result = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                width=w,
                height=h,
                guidance=guidance_scale,
                scheduler="linear",
                image_path=image_path,
                num_inference_steps=steps,
                image_strength=denoise_strength if image_path else None,
                negative_prompt=negative_prompt,
            )
        finally:
            if temp_file is not None:
                temp_file.close()
                Path(temp_file.name).unlink(missing_ok=True)

        elapsed = time.perf_counter() - start

        logger.info(
            "generation_complete",
            prompt_trunc=prompt[:60],
            seed=result.seed,
            elapsed=f"{elapsed:.2f}s",
        )

        return result.image, result.seed, elapsed

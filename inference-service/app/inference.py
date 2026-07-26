"""mflux inference wrapper.

Supports FLUX.1, FLUX.2, Z-Image, and other mflux model families.
Model is reloaded when LoRA paths change (MLX streams are thread-local,
so the model must be loaded on the worker thread).

This module is the one piece that changes when you swap the inference
backend. The rest of the service (routers, job queue) talks only to
the `InferenceEngine` interface.
"""

from __future__ import annotations

import base64
import io
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image

from app.config import settings

logger = structlog.get_logger(__name__)

# Models whose architectures support LoRA training.
# Derived from which mflux model directories contain a training_adapter/ module.
TRAINING_CAPABLE_MODELS = {
    # FLUX.2 Klein family
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-9b-kv",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    # Z-Image family
    "z-image",
    "z-image-turbo",
    # ERNIE-Image family (baidu/ERNIE-Image weights, Flux1-compatible arch)
    "ernie-image",
    "ernie-image-turbo",
}


def is_training_capable(model_alias: str) -> bool:
    """Check if a model alias supports LoRA training."""
    return model_alias in TRAINING_CAPABLE_MODELS


def _load_model_instance(model_alias: str, quantize: int | None, lora_paths: list[str] | None):
    """Dynamically import and construct the right model class for the given alias.

    Each mflux model family lives in its own package under ``mflux.models.*``.
    The constructor signatures vary: most accept ``(quantize, model_path,
    lora_paths, lora_scales, model_config)`` but some (FiboVLM, SeedVR2)
    use a different contract.
    """
    from mflux.models.common.config import ModelConfig

    model_config = ModelConfig.from_name(model_name=model_alias, base_model=None)
    lora_scales = [0.8] * len(lora_paths) if lora_paths else None

    # ── FLUX.2 Klein family ─────────────────────────────────
    if model_alias.startswith("flux2"):
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        return Flux2Klein(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        ), model_config

    # ── FLUX.1 family (dev, schnell, krea-dev) ──────────────
    if model_alias in ("dev", "schnell", "krea-dev", "dev-krea"):
        from mflux.models.flux.variants.txt2img.flux import Flux1

        return Flux1(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        ), model_config

    # ── Z-Image family ──────────────────────────────────────
    if model_alias.startswith("z-image"):
        from mflux.models.z_image.variants.z_image import ZImage

        return ZImage(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        ), model_config

    # ── ERNIE-Image family ──────────────────────────────────
    if model_alias.startswith("ernie-image"):
        from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

        return ErnieImage(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        ), model_config

    # ── Ideogram 4 ──────────────────────────────────────────
    if model_alias.startswith("ideogram"):
        from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4

        return Ideogram4(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        ), model_config

    # ── Qwen Image family ───────────────────────────────────
    if model_alias.startswith("qwen-image-edit"):
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        return QwenImageEdit(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        ), model_config

    if model_alias.startswith("qwen"):
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        ), model_config

    # ── FIBO VLM (image editing / VLM) — no LoRA support ───
    if model_alias.startswith("fibo"):
        from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

        return FiboVLM(
            model_id=None,
            model_path=None,
            quantize=quantize,
        ), model_config

    # ── SeedVR (upscaling) — no LoRA support ────────────────
    if model_alias.startswith("seedvr"):
        from mflux.models.seedvr2 import SeedVR2

        return SeedVR2(
            model_config=model_config,
            quantize=quantize,
            model_path=None,
        ), model_config

    # ── Default catch-all: try FLUX.1 ───────────────────────
    from mflux.models.flux.variants.txt2img.flux import Flux1

    return Flux1(
        model_config=model_config,
        quantize=quantize,
        model_path=None,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    ), model_config


class InferenceEngine:
    """Wraps an mflux model for local inference on Apple Silicon."""

    def __init__(self) -> None:
        self._model = None
        self._model_alias = settings.model_alias
        self._quantize = settings.quantize
        self._loaded_lora_paths: tuple[str, ...] = ()
        self._cancel_event = threading.Event()
        self._progress: float = 0.0  # 0.0 → 1.0, updated during generation
        self._num_steps: int = 0  # total steps for this generation

        # Apply resource limits once at init
        self._apply_resource_limits()

        logger.info(
            "inference_init",
            model=self._model_alias,
            quantize=self._quantize,
        )

    def cancel(self) -> None:
        """Signal the current generation to stop at the next diffusion step."""
        self._cancel_event.set()
        logger.info("cancel_signalled")

    @staticmethod
    def _apply_resource_limits() -> None:
        """Throttle MLX and OS to prevent Mac from freezing during inference/training."""
        import os as _os

        import mlx.core as mx

        # Lower CPU scheduling priority so the UI stays responsive
        try:
            _os.nice(settings.nice_level)
        except Exception:
            pass

        limit_gb = settings.mlx_cache_limit_gb or 8.0

        # Hard cap on total MLX memory (not just cache). Prevents OOM kernel panics.
        try:
            mx.set_memory_limit(int(limit_gb * 1024 * 1024 * 1024))
            logger.info("mlx_memory_limit_set", gb=limit_gb)
        except Exception as exc:
            logger.warning("mlx_memory_limit_failed", error=str(exc))

        # Limit MLX GPU memory cache
        try:
            mx.set_cache_limit(int(limit_gb * 1024 * 1024 * 1024))
            logger.info("mlx_cache_limit_set", gb=limit_gb)
        except Exception as exc:
            logger.warning("mlx_cache_limit_failed", error=str(exc))

        # Set wired memory limit as well (prevents system pressure)
        try:
            # Use 75% of cache limit as wired limit
            wired = int(limit_gb * 0.75 * 1024 * 1024 * 1024)
            mx.set_wired_limit(wired)
        except Exception:
            pass

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self, lora_paths: Optional[list[str]] = None) -> None:
        """Ensure the model is loaded with the given LoRA paths."""
        lora_tuple = tuple(lora_paths or [])
        if self._model is not None and self._loaded_lora_paths == lora_tuple:
            return

        logger.info(
            "model_loading",
            model=self._model_alias,
            quantize=self._quantize,
            loras=lora_tuple,
        )

        self._model, _ = _load_model_instance(
            model_alias=self._model_alias,
            quantize=self._quantize,
            lora_paths=list(lora_tuple) if lora_tuple else None,
        )
        # Register per-step cancel callback
        self._register_cancel_callback()
        self._loaded_lora_paths = lora_tuple
        logger.info("model_loaded", loras=self._loaded_lora_paths)

    def _register_cancel_callback(self) -> None:
        """Register an in-loop callback for cancel + progress tracking."""

        class _CancelChecker:
            def __init__(self, event: threading.Event, engine: "InferenceEngine"):
                self._event = event
                self._engine = engine

            def call_in_loop(self, t: int, **kwargs) -> None:
                # t is current step (0-indexed); total steps from engine state
                total = self._engine._num_steps
                self._engine._progress = (t + 1) / total if total > 0 else 0.0
                if self._event.is_set():
                    from mflux.utils.exceptions import StopImageGenerationException

                    raise StopImageGenerationException("Cancelled by user")

        self._cancel_event.clear()
        self._model.callbacks.register(_CancelChecker(self._cancel_event, self))

    def unload_model(self) -> None:
        self._model = None
        self._loaded_lora_paths = ()
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
        # Reset cancel so a previous cancellation doesn't leak through
        self._cancel_event.clear()

        self.load_model(lora_paths=lora_paths)

        steps = num_inference_steps or settings.default_num_steps
        h = height or settings.default_height
        w = width or settings.default_width
        seed = seed or int(time.time() * 1000) % (2**31)

        # FLUX.2 uses a different scheduler and doesn't support negative_prompt
        is_flux2 = self._model_alias.startswith("flux2")
        scheduler = "flow_match_euler_discrete" if is_flux2 else "linear"

        # ── img2img: save base64 ref to temp file ────────────
        image_path: Path | None = None
        temp_file = None
        if reference_image_b64 is not None:
            raw = base64.b64decode(reference_image_b64)
            ref = Image.open(io.BytesIO(raw)).convert("RGB")
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            ref.save(temp_file, format="PNG")
            image_path = Path(temp_file.name)

        # Reset progress for the new generation
        self._progress = 0.0
        self._num_steps = steps

        start = time.perf_counter()

        try:
            kwargs = {
                "seed": seed,
                "prompt": prompt,
                "width": w,
                "height": h,
                "guidance": guidance_scale,
                "scheduler": scheduler,
                "image_path": image_path,
                "num_inference_steps": steps,
                "image_strength": denoise_strength if image_path else None,
            }
            if not is_flux2 and negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            result = self._model.generate_image(**kwargs)
        finally:
            if temp_file is not None:
                temp_file.close()
                Path(temp_file.name).unlink(missing_ok=True)

        elapsed = time.perf_counter() - start

        logger.info(
            "generation_complete",
            prompt_trunc=prompt[:60],
            seed=result.seed,
            steps=steps,
            elapsed=f"{elapsed:.2f}s",
        )

        return result.image, result.seed, elapsed

    # ── Model info ───────────────────────────────────────────

    @property
    def loaded_lora_paths(self) -> list[str]:
        return list(self._loaded_lora_paths)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def progress(self) -> float:
        """Current generation progress (0.0 → 1.0)."""
        return self._progress

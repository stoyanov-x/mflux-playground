"""Pydantic schemas for the inference API contract.

This is the contract you'd re-implement against a cloud inference target
(Replicate, Fal.ai, RunPod) when scaling beyond the local Mac node.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════


class JobStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# ═══════════════════════════════════════════════════════════════
#  Generation Requests
# ═══════════════════════════════════════════════════════════════


class GenerationRequest(BaseModel):
    """Payload for a single image generation job."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = None

    # LoRA control — multiple LoRAs with individual scales
    lora_paths: Optional[list[str]] = None
    lora_scales: Optional[list[float]] = None

    # Image-to-image (img2img)
    reference_image_b64: Optional[str] = None
    denoise_strength: float = Field(default=0.4, ge=0.0, le=1.0)

    # Generation params
    num_inference_steps: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    seed: Optional[int] = None
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)

    # Model override (defaults to service config)
    model: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
#  Training Requests / Responses
# ═══════════════════════════════════════════════════════════════


class TrainingRequest(BaseModel):
    """Payload to start a LoRA training job."""

    name: str = Field(
        ..., min_length=1, max_length=100, description="LoRA name for the output file"
    )
    images: list[str] = Field(
        ..., min_length=3, max_length=100, description="Base64-encoded training images"
    )
    prompts: Optional[list[str]] = Field(
        None, description="Per-image prompts (optional; uses captions if omitted)"
    )

    # Training hyperparameters (optional, use defaults if omitted)
    model: Optional[str] = None
    steps: Optional[int] = None
    num_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    max_resolution: int = 1024

    @classmethod
    def validate_images(cls, images: list[str]) -> list[str]:
        """Ideal training set size: 10–30 high-quality images."""
        return images


# ═══════════════════════════════════════════════════════════════
#  Responses
# ═══════════════════════════════════════════════════════════════


class JobResponse(BaseModel):
    """Returned immediately when a job is submitted."""

    job_id: str
    status: JobStatus
    message: str = "Job accepted"


class JobStatusResponse(BaseModel):
    """Polling response for a generation job."""

    job_id: str
    status: JobStatus
    progress: float = 0.0  # 0.0 → 1.0
    result_url: Optional[str] = None
    error: Optional[str] = None
    seed_used: Optional[int] = None
    elapsed_seconds: Optional[float] = None


# ═══════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    status: str = "ok"
    model_alias: str = ""
    models_available: list[str] = []
    lora_loaded: list[str] = []
    quantize: Optional[int] = None

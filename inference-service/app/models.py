"""Pydantic schemas for the inference API contract.

This is the contract you'd re-implement against a cloud inference target
(Replicate, Fal.ai, RunPod) when scaling beyond the local Mac node.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


# ═══════════════════════════════════════════════════════════════
#  Requests
# ═══════════════════════════════════════════════════════════════

class GenerationRequest(BaseModel):
    """Payload for a single image generation job."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = None

    # LoRA control
    lora_paths: Optional[list[str]] = None
    lora_scales: Optional[list[float]] = None

    # Image-to-image (img2img)
    reference_image_b64: Optional[str] = None
    denoise_strength: float = Field(default=0.75, ge=0.0, le=1.0)

    # Generation params
    num_inference_steps: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    seed: Optional[int] = None
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0)


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
    progress: float = 0.0          # 0.0 → 1.0
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
    lora_loaded: list[str] = []
    quantize: Optional[int] = None

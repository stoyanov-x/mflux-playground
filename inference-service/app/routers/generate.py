"""Generation and training endpoints."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import Response

from app.models import (
    GenerationRequest,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    TrainingRequest,
)
from app.services.job_store import JobStore, get_job_store
from app.services.job_worker import get_worker

router = APIRouter(prefix="/api/v1", tags=["generation"])


# ═══════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════


@router.post("/generate", response_model=JobResponse, status_code=202)
async def submit_generation(
    req: GenerationRequest,
    store: JobStore = Depends(get_job_store),
    worker=Depends(get_worker),
) -> JobResponse:
    """Submit an image generation job with optional LoRAs and img2img."""
    job_id = str(uuid.uuid4())
    await asyncio.to_thread(store.create_job, job_id, req.model_dump())
    worker.enqueue(job_id)
    return JobResponse(job_id=job_id, status=JobStatus.queued)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> JobStatusResponse:
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/jobs/{job_id}/image")
async def get_job_image(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> Response:
    """Download the generated image for a completed job."""
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JobStatus.completed.value:
        raise HTTPException(status_code=400, detail="Job not yet completed")
    image_path = await asyncio.to_thread(store.get_image_path, job_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=Path(image_path).read_bytes(), media_type="image/png")


@router.get("/jobs")
async def list_jobs(
    store: JobStore = Depends(get_job_store),
) -> list[dict]:
    """List recent generation jobs (newest first)."""
    return await asyncio.to_thread(store.list_jobs, 50)


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> dict:
    """Delete a job and its image."""
    deleted = await asyncio.to_thread(store.delete_job, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> dict:
    """Cancel a queued or running job."""
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Job already {job['status']}")
    await asyncio.to_thread(store.update_job, job_id, status="cancelled")
    return {"status": "cancelled"}


# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════


@router.post("/train", response_model=JobResponse, status_code=202)
async def submit_training(
    req: TrainingRequest,
    store: JobStore = Depends(get_job_store),
    worker=Depends(get_worker),
) -> JobResponse:
    """Submit a LoRA training job.

    Accepts 3–100 base64-encoded images. 10–30 is the ideal range for
    a style LoRA. Training runs asynchronously — poll ``GET /jobs/{id}``
    for completion.
    """
    if len(req.images) < 3:
        raise HTTPException(
            status_code=422, detail="At least 3 training images are required"
        )
    if req.prompts and len(req.prompts) != len(req.images):
        raise HTTPException(
            status_code=422,
            detail=f"Number of prompts ({len(req.prompts)}) must match number of images ({len(req.images)})",
        )

    job_id = str(uuid.uuid4())
    # Tag as training job so the worker handles it differently
    await asyncio.to_thread(
        store.create_job, job_id, {"type": "train", **req.model_dump()}
    )
    worker.enqueue(job_id)
    return JobResponse(job_id=job_id, status=JobStatus.queued)


@router.get("/lora/list")
async def list_loras() -> list[dict]:
    """List trained LoRA adapters available for generation."""
    from app.services.training import TRAINING_DIR

    loras = []
    if TRAINING_DIR.exists():
        for f in sorted(TRAINING_DIR.glob("*.safetensors")):
            loras.append(
                {
                    "name": f.stem,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                }
            )
    return loras

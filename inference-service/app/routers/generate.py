"""Generation endpoints — the API contract the web backend calls."""

from __future__ import annotations

import io
import base64
import uuid
import time

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import Response

from app.models import GenerationRequest, JobResponse, JobStatus, JobStatusResponse
from app.services.job_store import JobStore, get_job_store
from app.services.job_worker import get_worker

router = APIRouter(prefix="/api/v1", tags=["generation"])


@router.post("/generate", response_model=JobResponse, status_code=202)
async def submit_generation(
    req: GenerationRequest,
    store: JobStore = Depends(get_job_store),
    worker=Depends(get_worker),
) -> JobResponse:
    """Submit an image generation job.

    Returns immediately with a ``job_id``. Poll ``GET /jobs/{id}``
    for the result.
    """
    job_id = str(uuid.uuid4())
    store.create_job(job_id, req.model_dump())
    worker.enqueue(job_id)
    return JobResponse(job_id=job_id, status=JobStatus.queued)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> JobStatusResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/jobs/{job_id}/image")
async def get_job_image(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> Response:
    """Download the generated image for a completed job."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JobStatus.completed.value:
        raise HTTPException(status_code=400, detail="Job not yet completed")
    image_bytes = store.get_image(job_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=image_bytes, media_type="image/png")

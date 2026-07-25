"""FastAPI application entrypoint for the mflux inference service.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import HealthResponse
from app.routers import generate
from app.services.job_worker import get_worker

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="mflux Inference Service",
    version="0.1.0",
    description="Local Flux inference via mflux on Apple Silicon. "
    "Designed as a swappable backend — implement the same API contract "
    "against a cloud target (Replicate, Fal.ai, RunPod) to scale.",
)

# Allow the web API to reach us (tighten in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router)


@app.on_event("startup")
async def preload_model():
    """Pre-load the model in the background so download progress is visible."""
    async def _background_load():
        logger.info("Pre-loading model in background (download will show below)…")

        def _load():
            from app.services.job_worker import get_worker
            worker = get_worker()
            worker._engine.load_model()
            logger.info("Model pre-loaded successfully.")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load)

    asyncio.create_task(_background_load())


@app.get("/health", response_model=HealthResponse)
async def health():
    from app.services.job_worker import get_worker

    try:
        worker = get_worker()
        model_ready = worker._engine.is_loaded
        lora_loaded = worker._engine.loaded_lora_paths
    except Exception:
        model_ready = False
        lora_loaded = []

    return HealthResponse(
        status="ok" if model_ready else "loading",
        model_alias=settings.model_alias,
        models_available=[
            "flux2-klein-4b",
            "flux2-klein-9b",
            "z-image-turbo",
            "z-image",
        ],
        lora_loaded=lora_loaded,
        quantize=settings.quantize,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=True,
    )

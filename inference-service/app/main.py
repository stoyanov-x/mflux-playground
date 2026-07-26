"""FastAPI application entrypoint for the mflux inference service.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import HealthResponse
from app.routers import generate

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


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_alias=settings.model_alias,
        models_available=[
            "flux2-klein-4b",
            "flux2-klein-9b",
            "z-image-turbo",
            "z-image",
        ],
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
        # uvicorn removed this parameter; enforced via middleware instead
        # limit_max_request_body_size=settings.max_request_body_size_mb * 1024 * 1024,
    )

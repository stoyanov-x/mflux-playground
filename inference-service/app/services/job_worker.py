"""Simple background worker that processes generation jobs.

Picks up jobs from the in-memory store and runs them through
the mflux ``InferenceEngine`` one at a time. This keeps the
FastAPI event loop responsive and enforces the single-node
concurrency constraint (M3 Max handles one gen at a time).
"""

from __future__ import annotations

import io
import threading
import time

import structlog
from mflux.utils.exceptions import StopImageGenerationException

from app.config import settings
from app.inference import InferenceEngine
from app.models import JobStatus
from app.services.job_store import JobStore, get_job_store

logger = structlog.get_logger(__name__)


class JobWorker:
    """Background thread that drains the job queue sequentially."""

    def __init__(self, engine: InferenceEngine, store: JobStore) -> None:
        self._engine = engine
        self._store = store
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Model is loaded lazily by the worker thread (MLX streams are thread-local).
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("worker_started")

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            self._queue.append(job_id)
            self._cv.notify()

    def _run(self) -> None:
        while True:
            job_id = self._dequeue()
            if job_id is None:
                break
            self._process(job_id)

    def _dequeue(self) -> str | None:
        with self._lock:
            while not self._queue:
                self._cv.wait()
            return self._queue.pop(0)

    def _process(self, job_id: str) -> None:
        try:
            job = self._store.get_job(job_id)
            if job is None:
                return

            # Check if cancelled while queued
            if job.get("status") == "cancelled":
                logger.info("job_cancelled_while_queued", job_id=job_id)
                return

            self._store.update_job(job_id, status=JobStatus.running.value, progress=0.0)
            logger.info("job_started", job_id=job_id, type=job.get("type", "generate"))

            try:
                req = self._store.get_params(job_id) or {}

                # Dispatch: training vs generation
                if req.get("type") == "train":
                    self._run_training(job_id, req)
                else:
                    self._run_generation(job_id, req)

            except StopImageGenerationException:
                logger.info("job_cancelled", job_id=job_id)
                self._store.update_job(
                    job_id,
                    status=JobStatus.cancelled.value,
                )
            except Exception as exc:
                logger.exception("job_failed", job_id=job_id, error=str(exc))
                self._store.update_job(
                    job_id,
                    status=JobStatus.failed.value,
                    error=str(exc),
                )
        except BaseException as exc:
            # Last-resort catch — prevents the entire worker from dying
            logger.exception("worker_crashed", job_id=job_id, error=str(exc))

    def _run_generation(self, job_id: str, req: dict) -> None:
        # Log warning if a different model was requested (not supported per-request)
        req_model = req.get("model")
        if req_model and req_model != self._engine._model_alias:
            logger.warning(
                "model_override_ignored",
                requested=req_model,
                active=self._engine._model_alias,
            )

        # Signal "downloading" before model load, "running" after
        self._store.update_job(job_id, status=JobStatus.downloading.value, progress=0.0)

        # Lazy-load model on the worker thread (MLX GPU streams are thread-local)
        self._engine.load_model(lora_paths=req.get("lora_paths"))

        self._store.update_job(job_id, status=JobStatus.running.value, progress=0.0)

        # Reset progress on the engine *before* the poller starts
        self._engine._progress = 0.0

        # Start a daemon thread to poll progress during generation
        def _poll_progress() -> None:
            while True:
                p = self._engine.progress
                if p > 0.0:
                    self._store.update_job(job_id, progress=p)
                time.sleep(0.5)

        poller = threading.Thread(target=_poll_progress, daemon=True)
        poller.start()

        try:
            image, seed, elapsed = self._engine.generate(
                prompt=req.get("prompt", ""),
                negative_prompt=req.get("negative_prompt"),
                lora_paths=req.get("lora_paths"),
                lora_scales=req.get("lora_scales"),
                reference_image_b64=req.get("reference_image_b64"),
                denoise_strength=req.get("denoise_strength", 0.4),
                num_inference_steps=req.get("num_inference_steps"),
                height=req.get("height"),
                width=req.get("width"),
                seed=req.get("seed"),
                guidance_scale=req.get("guidance_scale", 3.5),
            )
        finally:
            # Ensure progress is finalised even if generation errors
            self._store.update_job(job_id, progress=1.0)

        # Save to disk and store path in DB
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        image_path = self._store.set_image(job_id, image_bytes)
        result_url = f"/api/v1/jobs/{job_id}/image"

        self._store.update_job(
            job_id,
            status=JobStatus.completed.value,
            progress=1.0,
            result_url=result_url,
            image_path=image_path,
            seed_used=seed,
            elapsed_seconds=round(elapsed, 2),
        )
        logger.info(
            "job_completed",
            job_id=job_id,
            path=image_path,
            elapsed=f"{elapsed:.2f}s",
        )

    def _run_training(self, job_id: str, req: dict) -> None:
        from app.services.training import run_training

        output_path = run_training(
            name=req.get("name", f"lora-{job_id[:8]}"),
            images_b64=req.get("images", []),
            prompts=req.get("prompts"),
            model_alias=req.get("model"),
            steps=req.get("steps"),
            num_epochs=req.get("num_epochs"),
            learning_rate=req.get("learning_rate"),
            max_resolution=req.get("max_resolution", settings.training_max_resolution),
        )

        self._store.update_job(
            job_id,
            status=JobStatus.completed.value,
            progress=1.0,
            result_url="/api/v1/lora/list",
            seed_used=None,
            elapsed_seconds=None,
        )
        logger.info("training_completed", job_id=job_id, lora_path=str(output_path))


# ── Dependency ──────────────────────────────────────────────

_worker_instance: JobWorker | None = None


def get_worker() -> JobWorker:
    global _worker_instance
    if _worker_instance is None:
        engine = InferenceEngine()
        store = get_job_store()
        _worker_instance = JobWorker(engine, store)
        _worker_instance.start()
    return _worker_instance

"""In-memory job store.

Keeps track of generation jobs and their results.
Replace with Redis / Postgres when adding persistence or scaling
to multiple worker nodes.
"""

from __future__ import annotations

import threading
from typing import Optional

from app.config import settings
from app.models import JobStatus


class JobStore:
    """Thread-safe in-memory store for generation jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._images: dict[str, bytes] = {}

    # ── CRUD ─────────────────────────────────────────────────

    def create_job(self, job_id: str, params: dict) -> dict:
        record = {
            "job_id": job_id,
            "status": JobStatus.queued.value,
            "progress": 0.0,
            "result_url": None,
            "error": None,
            "seed_used": None,
            "elapsed_seconds": None,
            "params": params,
        }
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            raw = self._jobs.get(job_id)
            if raw is None:
                return None
            # Return a copy sans internal params
            return {k: v for k, v in raw.items() if k != "params"}

    def update_job(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def set_image(self, job_id: str, image_bytes: bytes) -> None:
        with self._lock:
            self._images[job_id] = image_bytes
            # Enforce cap
            if len(self._images) > settings.max_completed_jobs:
                oldest = next(iter(self._images))
                del self._images[oldest]

    def get_image(self, job_id: str) -> Optional[bytes]:
        with self._lock:
            return self._images.get(job_id)


# ── Dependency ──────────────────────────────────────────────

_store_instance: JobStore | None = None


def get_job_store() -> JobStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = JobStore()
    return _store_instance

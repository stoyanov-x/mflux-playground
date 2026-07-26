"""SQLite-backed job store + disk-backed images.

Jobs metadata lives in SQLite; generated images are saved as PNG files
on disk with a path reference in the database.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

from app.config import settings
from app.models import JobStatus

# Database file lives alongside the app package (gitignored).
_DEFAULT_DB_DIR = Path(__file__).parent.parent.parent / "data"
_IMAGES_DIR = Path(__file__).parent.parent.parent / "generated_images"


class JobStore:
    """SQLite-persisted store for generation jobs + disk-backed images."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_dir = _DEFAULT_DB_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "mflux_jobs.db")
        self._db_path = db_path
        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._max_images = settings.max_completed_jobs

    # ── Connection helpers ───────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id        TEXT PRIMARY KEY,
                    status        TEXT NOT NULL DEFAULT 'queued',
                    progress      REAL NOT NULL DEFAULT 0.0,
                    result_url    TEXT,
                    error         TEXT,
                    seed_used     INTEGER,
                    elapsed_seconds REAL,
                    params_json   TEXT,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

                -- Migration: drop old BLOB table if upgrading from earlier schema
                DROP TABLE IF EXISTS images;
            """)
            # Migration: add image_path column if missing (idempotent)
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE jobs ADD COLUMN image_path TEXT")

    # ── CRUD ─────────────────────────────────────────────────

    def create_job(self, job_id: str, params: dict) -> dict:
        record = {
            "job_id": job_id,
            "status": JobStatus.queued.value,
            "progress": 0.0,
            "result_url": None,
            "image_path": None,
            "error": None,
            "seed_used": None,
            "elapsed_seconds": None,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, params_json) VALUES (?, ?, ?)",
                (job_id, JobStatus.queued.value, json.dumps(params)),
            )
        return record

    def get_job(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT job_id, status, progress, result_url, image_path, error,
                          seed_used, elapsed_seconds
                   FROM jobs WHERE job_id = ?""",
                (job_id,),
            ).fetchone()
            return dict(row) if row else None

    def update_job(self, job_id: str, **fields: object) -> None:
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = [*fields.values(), job_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE job_id = ?",
                values,
            )

    def get_params(self, job_id: str) -> dict | None:
        """Retrieve the original request params stored at creation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT params_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row and row["params_json"]:
                return json.loads(row["params_json"])
            return None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return recent jobs, newest first, with prompt extracted from params."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT job_id, status, progress, result_url, image_path, error,
                          seed_used, elapsed_seconds, created_at, params_json
                   FROM jobs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            result = []
            for r in rows:
                job = dict(r)
                # Extract prompt from params_json
                params = json.loads(job.pop("params_json") or "{}")
                job["prompt"] = params.get("prompt") or params.get("name") or "[training job]"
                result.append(job)
            return result

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated image file. Returns True if deleted."""
        # Get image path first
        image_path = self.get_image_path(job_id)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            deleted = cursor.rowcount > 0
        # Remove image file
        if image_path and Path(image_path).exists():
            Path(image_path).unlink(missing_ok=True)
        return deleted

    # ── Disk-backed images ───────────────────────────────────

    def set_image(self, job_id: str, image_bytes: bytes) -> str:
        """Save image bytes to disk and return a portable relative path."""
        image_path = _IMAGES_DIR / f"{job_id}.png"
        image_path.write_bytes(image_bytes)
        # Store only the filename so the path stays valid if the project moves.
        return image_path.name  # e.g. "uuid.png"

    def get_image_path(self, job_id: str) -> str | None:
        """Return the on-disk path for a completed job's image, or None.

        Resolves stored paths dynamically so images are found even when
        the project directory has moved (old absolute paths stored by
        earlier versions of the code still work via filename fallback).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_path FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            stored = row["image_path"] if row else None
            if not stored:
                return None

            candidate = Path(stored)
            if candidate.is_absolute():
                # Old-style absolute path — check if it still exists
                if candidate.exists():
                    return str(candidate)
                # Fall back to the filename in the current images directory
                candidate = _IMAGES_DIR / candidate.name

            else:
                # New-style relative path (just the filename)
                candidate = _IMAGES_DIR / stored

            return str(candidate) if candidate.exists() else None


# ── Dependency ──────────────────────────────────────────────

_store_instance: JobStore | None = None


def get_job_store() -> JobStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = JobStore()
    return _store_instance

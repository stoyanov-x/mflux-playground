"""SQLite-backed job store.

Persists jobs and generated images so they survive service restarts.
Swap for Postgres / Redis when scaling to multi-worker deployments.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models import JobStatus

# Database file lives alongside the app package (gitignored).
_DEFAULT_DB_DIR = Path(__file__).parent.parent.parent / "data"


class JobStore:
    """SQLite-persisted store for generation jobs.

    Creates tables on init. Thread-safe via per-call connections
    with WAL mode and busy timeout for concurrency.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_dir = _DEFAULT_DB_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "mflux_jobs.db")
        self._db_path = db_path
        self._init_db()
        # Cap for images (oldest removed when exceeded)
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

                CREATE TABLE IF NOT EXISTS images (
                    job_id  TEXT PRIMARY KEY,
                    data    BLOB NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
            """)

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
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, params_json) VALUES (?, ?, ?)",
                (job_id, JobStatus.queued.value, json.dumps(params)),
            )
        return record

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT job_id, status, progress, result_url, error,
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
                f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values,
            )

    def get_params(self, job_id: str) -> Optional[dict]:
        """Retrieve the original request params stored at creation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT params_json FROM jobs WHERE job_id = ?", (job_id,),
            ).fetchone()
            if row and row["params_json"]:
                return json.loads(row["params_json"])
            return None

    # ── Image BLOBs ──────────────────────────────────────────

    def set_image(self, job_id: str, image_bytes: bytes) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO images (job_id, data) VALUES (?, ?)",
                (job_id, image_bytes),
            )
            # Enforce cap — remove oldest image rows beyond limit.
            conn.execute(
                """DELETE FROM images WHERE job_id IN (
                       SELECT job_id FROM images
                       ORDER BY job_id ASC
                       LIMIT MAX(0, (SELECT COUNT(*) FROM images) - ?)
                   )""",
                (self._max_images,),
            )

    def get_image(self, job_id: str) -> Optional[bytes]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM images WHERE job_id = ?", (job_id,),
            ).fetchone()
            return row["data"] if row else None


# ── Dependency ──────────────────────────────────────────────

_store_instance: JobStore | None = None


def get_job_store() -> JobStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = JobStore()
    return _store_instance

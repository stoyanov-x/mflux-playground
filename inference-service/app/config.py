"""Configuration for the mflux inference service.

Uses pydantic-settings so all values can be overridden via env vars.
This keeps the inference contract portable — swap the backend without
changing any calling code.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class InferenceSettings(BaseSettings):
    """Settings loaded from environment / .env file."""

    # ── Serving ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ── Model defaults ───────────────────────────────────────
    model_alias: str = "dev"               # "dev" | "schnell"
    default_num_steps: int = 25
    default_height: int = 1024
    default_width: int = 1024

    # ── Quantisation ─────────────────────────────────────────
    # Set to 4 or 8 to enable on-the-fly quantisation.
    quantize: Optional[int] = None

    # ── LoRA ─────────────────────────────────────────────────
    # Comma-separated paths to default LoRA weight files.
    lora_paths: str = ""

    # ── Job store ────────────────────────────────────────────
    max_completed_jobs: int = 100

    model_config = {"env_prefix": "MFLUX_"}


settings = InferenceSettings()

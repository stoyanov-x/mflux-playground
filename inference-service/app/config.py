"""Configuration for the mflux inference service.

Uses pydantic-settings so all values can be overridden via env vars.
This keeps the inference contract portable — swap the backend without
changing any calling code.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class InferenceSettings(BaseSettings):
    """Settings loaded from environment / .env file."""

    # ── Serving ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ── Model defaults ───────────────────────────────────────
    # Best models for Mac + LoRA training (see README for full list):
    #   "flux2-klein-4b" — fastest, smallest, supports training
    #   "z-image-turbo"  — fast, good quality, supports training
    #   "flux2-klein-9b" — higher quality, supports training
    model_alias: str = "flux2-klein-4b"
    base_model: Optional[str] = None
    default_num_steps: int = 25
    default_height: int = 1024
    default_width: int = 1024

    # ── Quantisation ─────────────────────────────────────────
    # Set to 4 or 8 to enable on-the-fly quantisation.
    quantize: Optional[int] = 8

    # ── LoRA ─────────────────────────────────────────────────
    # Comma-separated paths to default LoRA weight files.
    lora_paths: str = ""

    # ── Job store ────────────────────────────────────────────
    max_completed_jobs: int = 100

    # ── Resource limits (prevents Mac from freezing) ─────────
    # Limit MLX GPU memory cache in GB (e.g. 8, 16). None = unlimited.
    mlx_cache_limit_gb: Optional[float] = 8.0
    # Low RAM mode: offloads training data to disk (saves ~10 GB at 1024px).
    low_ram: bool = True
    # CPU nice value (higher = lower priority, 0=normal, 10=low, 19=idle).
    nice_level: int = 10

    # ── Training ─────────────────────────────────────────────
    training_dir: str = "training"
    training_steps: int = 9
    training_epochs: int = 50
    training_learning_rate: float = 1e-4
    # Max image resolution for training (lower = less memory). 1024 works on
    # 64GB+ machines; use 768 for 48GB machines like M3 Max.
    training_max_resolution: int = 768

    # ── Body size ────────────────────────────────────────────
    # Max request body in MB (training uploads can be ~150 MB)
    max_request_body_size_mb: int = 500

    model_config = {"env_prefix": "MFLUX_", "env_file": ".env"}


settings = InferenceSettings()

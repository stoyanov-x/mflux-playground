"""LoRA training runner.

Wraps mflux's ``TrainingRunner`` to train a LoRA adapter from user-supplied
images. Runs synchronously on the background worker thread (one training
at a time, same as generation).
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import structlog
from PIL import Image

from app.config import settings

logger = structlog.get_logger(__name__)

# Training output directory
TRAINING_DIR = Path(settings.training_dir)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# Default training config template (FLUX.2 Klein / Z-Image compatible)
DEFAULT_LORA_TARGETS = [
    {
        "module_path": "layers.{block}.attention.to_q",
        "blocks": {"start": 0, "end": 18},
        "rank": 16,
    },
    {
        "module_path": "layers.{block}.attention.to_k",
        "blocks": {"start": 0, "end": 18},
        "rank": 16,
    },
    {
        "module_path": "layers.{block}.attention.to_v",
        "blocks": {"start": 0, "end": 18},
        "rank": 16,
    },
    {
        "module_path": "layers.{block}.attention.to_out.0",
        "blocks": {"start": 0, "end": 18},
        "rank": 16,
    },
    {
        "module_path": "layers.{block}.feed_forward.w1",
        "blocks": {"start": 0, "end": 18},
        "rank": 16,
    },
    {
        "module_path": "layers.{block}.feed_forward.w2",
        "blocks": {"start": 0, "end": 18},
        "rank": 16,
    },
    {
        "module_path": "layers.{block}.feed_forward.w3",
        "blocks": {"start": 0, "end": 18},
        "rank": 16,
    },
]


def build_training_config(
    *,
    model_alias: str,
    data_dir: str,
    output_path: str,
    steps: int,
    num_epochs: int,
    learning_rate: float,
    max_resolution: int,
    quantize: int | None,
    low_ram: bool = False,
) -> dict:
    """Build a training config dict matching mflux's ``train.json`` format."""
    return {
        "model": model_alias,
        "data": data_dir,
        "seed": int(time.time()) % (2**31),
        "steps": steps,
        "guidance": 0.0,
        "quantize": quantize,
        "max_resolution": max_resolution,
        "low_ram": low_ram,
        "training_loop": {
            "num_epochs": num_epochs,
            "batch_size": 1,
            "timestep_low": 4,
            "timestep_high": steps,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
        },
        "checkpoint": {
            "save_frequency": max(1, num_epochs // 3),
            "output_path": output_path,
        },
        "monitoring": {
            "preview_width": 1024,
            "preview_height": 1024,
            "plot_frequency": 1,
            "generate_image_frequency": max(1, num_epochs // 3),
        },
        "lora_layers": {
            "targets": DEFAULT_LORA_TARGETS,
        },
    }


def run_training(
    *,
    name: str,
    images_b64: list[str],
    prompts: list[str] | None,
    model_alias: str | None,
    steps: int | None,
    num_epochs: int | None,
    learning_rate: float | None,
    max_resolution: int,
) -> Path:
    """Run a full LoRA training cycle.

    Returns the path to the trained ``.safetensors`` file.
    """
    # 1. Set low CPU priority so the Mac stays usable during training
    try:
        os.nice(settings.nice_level)
    except Exception:
        pass

    # 2. Resolve params
    model = model_alias or settings.model_alias
    n_steps = steps or settings.training_steps
    n_epochs = num_epochs or settings.training_epochs
    lr = learning_rate or settings.training_learning_rate
    quant = settings.quantize

    # 3. Save images to a temp directory
    job_dir = TRAINING_DIR / name
    if job_dir.exists():
        shutil.rmtree(job_dir)
    data_dir = job_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)

    for i, b64_img in enumerate(images_b64):
        raw = base64.b64decode(b64_img)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(data_dir / f"{i:04d}.png")

    # 4. Write prompt captions if provided
    if prompts:
        captions_file = data_dir / "metadata.jsonl"
        with open(captions_file, "w") as f:
            for i, prompt in enumerate(prompts):
                json.dump({"image": f"{i:04d}.png", "prompt": prompt}, f)
                f.write("\n")

    # 5. Write training config
    config = build_training_config(
        model_alias=model,
        data_dir=str(data_dir),
        output_path=str(job_dir / "checkpoints"),
        steps=n_steps,
        num_epochs=n_epochs,
        learning_rate=lr,
        max_resolution=max_resolution,
        quantize=quant,
        low_ram=settings.low_ram,
    )
    config_path = job_dir / "train.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "training_start",
        name=name,
        model=model,
        steps=n_steps,
        epochs=n_epochs,
        images=len(images_b64),
    )

    # 5. Run training via mflux's TrainingRunner
    from mflux.models.common.training.runner import TrainingRunner

    try:
        adapter, spec = TrainingRunner.train(
            config_path=str(config_path), resume_path=None
        )

        # 6. Find the latest checkpoint and extract the LoRA safetensors
        checkpoints_dir = job_dir / "checkpoints"
        checkpoint_zips = sorted(checkpoints_dir.glob("*_checkpoint.zip"))
        if not checkpoint_zips:
            raise RuntimeError("No checkpoint zip found after training")

        latest_zip = checkpoint_zips[-1]
        import zipfile

        output_file = TRAINING_DIR / f"{name}.safetensors"
        with zipfile.ZipFile(latest_zip, "r") as zf:
            # Find the adapter safetensors inside the zip
            adapter_names = [
                n for n in zf.namelist() if n.endswith("_adapter.safetensors")
            ]
            if not adapter_names:
                raise RuntimeError(f"No adapter safetensors found in {latest_zip}")
            with zf.open(adapter_names[0]) as src, open(output_file, "wb") as dst:
                dst.write(src.read())

        logger.info("training_complete", name=name, output=str(output_file))
        return output_file
    except Exception:
        logger.exception("training_failed", name=name)
        raise

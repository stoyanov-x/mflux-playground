# Inference Service API Contract

This document defines the contract between the **web API** and the **inference service**.
The contract is designed to be backend-agnostic — the same endpoints work whether
you're running mflux locally on Apple Silicon or a cloud-hosted diffusers pipeline.

---

## Endpoints

### `POST /api/v1/generate`
Submit a generation job. Returns immediately with a `job_id`.

**Request body:**
```json
{
  "prompt": "a portrait in the style of...",
  "negative_prompt": null,
  "lora_paths": ["training/my-style.safetensors"],
  "lora_scales": [0.8],
  "reference_image_b64": null,
  "denoise_strength": 0.4,
  "num_inference_steps": 25,
  "height": 1024,
  "width": 1024,
  "seed": null,
  "guidance_scale": 3.5,
  "model": null
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "uuid-here",
  "status": "queued",
  "message": "Job accepted"
}
```

---

### `POST /api/v1/train`
Submit a LoRA training job. Accepts 3–100 base64-encoded images.

**Request body:**
```json
{
  "name": "my-art-style",
  "images": ["<base64-encoded-png>", "..."],
  "prompts": ["a portrait in this style", "a landscape in this style"],
  "model": "flux2-klein-4b",
  "steps": 9,
  "num_epochs": 50,
  "learning_rate": 0.0001,
  "max_resolution": 1024
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "uuid-here",
  "status": "queued",
  "message": "Job accepted"
}
```

---

### `GET /api/v1/jobs/{job_id}`
Poll for job status (works for both generation and training).

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "progress": 1.0,
  "result_url": "/api/v1/jobs/uuid-here/image",
  "error": null,
  "seed_used": 42,
  "elapsed_seconds": 23.45
}
```

Status values: `queued` → `running` → `completed` | `failed`

---

### `GET /api/v1/jobs/{job_id}/image`
Download the generated PNG (generation jobs only).

---

### `GET /api/v1/lora/list`
List trained LoRA adapters available for generation.

**Response:**
```json
[
  { "name": "my-art-style", "path": "training/my-art-style.safetensors", "size_bytes": 123456 }
]
```

---

### `GET /health`
Health check.

**Response:**
```json
{
  "status": "ok",
  "model_alias": "flux2-klein-4b",
  "models_available": ["flux2-klein-4b", "flux2-klein-9b", "z-image-turbo", "z-image"],
  "lora_loaded": [],
  "quantize": 8
}
```

---

## Supported Models

| Model | Alias | Training | Notes |
|---|---|---|---|
| FLUX.2 Klein 4B | `flux2-klein-4b` | ✅ Yes | Fastest, smallest, recommended default |
| FLUX.2 Klein 9B | `flux2-klein-9b` | ✅ Yes | Higher quality, larger |
| Z-Image Turbo | `z-image-turbo` | ✅ Yes | Fast, great realism, good for training |
| Z-Image | `z-image` | ✅ Yes | Base Z-Image |
| FLUX.1 dev | `dev` | ❌ No | Legacy, no training |
| FLUX.1 schnell | `schnell` | ❌ No | Legacy, no training |

---

## Cloud Backend Porting Checklist

When implementing this contract against a cloud target:

1. [ ] **`POST /api/v1/generate`** — Submit to cloud API (Replicate / Fal.ai / RunPod), return a tracking ID
2. [ ] **`POST /api/v1/train`** — Submit training to cloud GPU (RunPod / Lambda), return a tracking ID
3. [ ] **`GET /api/v1/jobs/{id}`** — Poll cloud API for status; map their states to our enum
4. [ ] **`GET /api/v1/jobs/{id}/image`** — Proxy/redirect to cloud-hosted image URL
5. [ ] **`GET /api/v1/lora/list`** — List trained LoRAs stored in cloud storage (S3, etc.)
6. [ ] **`GET /health`** — Return service status


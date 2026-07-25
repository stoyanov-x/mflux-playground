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
  "lora_paths": ["lora_weights/style.safetensors"],
  "lora_scales": [0.8],
  "reference_image_b64": null,
  "denoise_strength": 0.75,
  "num_inference_steps": 25,
  "height": 1024,
  "width": 1024,
  "seed": null,
  "guidance_scale": 3.5
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
Poll for job status.

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
Download the generated PNG.

---

### `GET /health`
Health check.

**Response:**
```json
{
  "status": "ok",
  "model_alias": "dev",
  "lora_loaded": [],
  "quantize": 8
}
```

---

## Cloud Backend Porting Checklist

When implementing this contract against a cloud target:

1. [ ] **`POST /api/v1/generate`** — Submit to cloud API (Replicate / Fal.ai / RunPod), return a tracking ID
2. [ ] **`GET /api/v1/jobs/{id}`** — Poll cloud API for status; map their states to our enum
3. [ ] **`GET /api/v1/jobs/{id}/image`** — Proxy/redirect to cloud-hosted image URL
4. [ ] **`GET /health`** — Return service status

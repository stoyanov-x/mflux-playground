# mflux Experiment

Flux image generation on Apple Silicon, with a cloud-portable architecture.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  React SPA   │────▶│  ASP.NET Core    │────▶│  FastAPI Inference  │
│  (Vite)      │     │  Web API         │     │  Service (mflux)    │
│  :5173       │◀────│  :5100           │◀────│  :8000              │
└──────────────┘     └──────────────────┘     └─────────────────────┘
                          │                           │
                     ┌────┴────┐              ┌──────┴──────┐
                     │ Job     │              │  mflux      │
                     │ Queue   │              │  (MLX)      │
                     │ (mem)   │              │  on M3 Max  │
                     └─────────┘              └─────────────┘
```

Three services, designed so the **inference backend is swappable** without touching the web API or frontend.

---

## Services

### 1. Inference Service (`inference-service/`)
- **FastAPI** wrapping **mflux** (MLX-native Flux implementation)
- Runs only on Apple Silicon (requires MLX)
- Single-threaded background worker — queues jobs sequentially
- Supports LoRA (multiple, stacked), img2img, quantisation
- Exposes a clean REST contract documented in `shared/API_CONTRACT.md`

### 2. Web API (`web-api/`)
- **ASP.NET Core 9** API with an in-memory job tracker
- `IInferenceClient` abstraction — swap the implementation to point at Replicate, Fal.ai, or RunPod
- CORS-enabled for local frontend dev

### 3. Frontend (`web-frontend/`)
- **React 19 + Vite + TypeScript**
- Three-tab SPA: Generate (with LoRA selector, img2img), Train LoRA, History
- In dev mode (`npm run dev`): Vite proxy forwards API calls to backend services
- In Docker: nginx serves static files and reverse-proxies API calls via `nginx.conf.template`

---

## Quick Start (Local Dev)

### ⚠️ Docker limitation on Mac

The inference service (mflux) uses **MLX**, which talks directly to Apple's **Metal GPU** framework. Docker for Mac runs inside a Linux VM — it has **no access to the Metal GPU**, so the inference service **cannot run in Docker on macOS**.

### Prerequisites
- Python 3.12+ with [`uv`](https://docs.astral.sh/uv/)
- Apple Silicon Mac (for mflux inference)
- Node.js 22+ *(only for frontend)*
- Docker Desktop *(optional — hybrid mode)*

### Quick start

```bash
./start.sh
```

This installs deps, starts the inference service natively, Docker containers for
web API + frontend, and opens http://localhost:5173.

**First run** will download the FLUX.2 Klein 4B model (~6 GB). This is cached
after the first download.

### Manual start (three terminals)

```bash
# Terminal 1 — inference
cd inference-service && uv run uvicorn app.main:app --reload --port 8000

# Terminal 2 — supporting services (Docker)
docker compose up web-api frontend

# Terminal 3 — frontend (if not using Docker)
cd web-frontend && npm install && npm run dev
```

---

## What you can do

### 🎨 Generate images

Three tabs in the web UI at http://localhost:5173:

**Generate tab** — prompt + negative prompt, select trained LoRAs with
per-adapter scale sliders, upload a reference image for img2img, adjust
steps and guidance.

**Train LoRA tab** — upload 10-30 images (drag & drop), name your LoRA,
configure epochs/steps, start training. Training takes 20-60 minutes on
M3 Max depending on image count and epochs. The trained `.safetensors`
automatically appears in the Generate tab's LoRA selector.

**History tab** — past generations with status badges and image previews.

### 🧠 Train a LoRA via API

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my-watercolor-style",
    "images": ["<base64-png>", "..."]
  }'
```

Then generate with it:

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a portrait in watercolor style",
    "lora_paths": ["training/my-watercolor-style.safetensors"],
    "lora_scales": [0.8]
  }'
```

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  React SPA   │────▶│  ASP.NET Core    │────▶│  FastAPI Inference      │
│  (Vite+TW)   │     │  Web API         │     │  Service (mflux)        │
│  :5173       │◀────│  :5100           │◀────│  :8000                  │
└──────────────┘     └──────────────────┘     │                         │
                          │                    │  ├ Generation worker    │
                     ┌────┴────┐              │  ├ Training runner      │
                     │ SQLite  │              │  ├ MLX cache limit 16GB │
                     │ (EF C)  │              │  ├ os.nice(10)          │
                     └─────────┘              │  └ LoRA store (files)   │
                                              └─────────────────────────┘
```

---

## Configuration

### Nginx proxy (Docker frontend)

When running the frontend in Docker, `nginx.conf.template` uses
`envsubst` at startup to inject two environment variables:

| Variable | Default (Mac) | Default (Linux) |
|---|---|---|
| `INFERENCE_API_URL` | `http://host.docker.internal:8000` | `http://inference-service:8000` |
| `WEB_API_URL` | `http://web-api:8080` | `http://web-api:8080` |

Override in `docker-compose.yml` or via `docker compose run -e`:

```yaml
frontend:
  environment:
    INFERENCE_API_URL: "http://inference-service:8000"
```

### Resource limits

Set via `.env` or environment variables (prefix `MFLUX_`):

- `MFLUX_MLX_CACHE_LIMIT_GB` — MLX GPU cache (default: 16 GB)
- `MFLUX_LOW_RAM` — offload training data to disk (default: true)
- `MFLUX_NICE_LEVEL` — CPU priority, higher = lower (default: 10)

---

## Supported Models

| Model | Alias | Training | Size | Notes |
|---|---|---|---|---|
| **FLUX.2 Klein 4B** | `flux2-klein-4b` | ✅ | 4B params | **Default.** Fastest, smallest, very good quality |
| FLUX.2 Klein 9B | `flux2-klein-9b` | ✅ | 9B params | Higher quality, larger |
| Z-Image Turbo | `z-image-turbo` | ✅ | 6B params | Great realism, fast |
| Z-Image | `z-image` | ✅ | 6B params | Base Z-Image |
| FLUX.1 dev/schnell | `dev` / `schnell` | ❌ | 12B params | Legacy, no training, removed from cache |

---

## Secret Scanning (TruffleHog)

TruffleHog runs automatically in CI on every push/PR to `main` and weekly as a full-history scan.

### Local usage

```bash
# Scan the whole repo
make trufflehog

# Or via pre-commit (scans staged files before push)
make setup-dev    # one-time setup — installs pre-commit + TruffleHog
git add . && git commit -m "..."   # pre-commit hooks run
git push                            # pre-push hook runs TruffleHog
```

The pre-push hook uses `--only-verified` to avoid noise. To bypass it for a false positive: `git push --no-verify`.

---

## Cloud Portability

The `IInferenceClient` interface in the web API is the swap point.
To move inference to the cloud:

1. Implement `IInferenceClient` against Replicate / Fal.ai / RunPod
2. Register it in `Program.cs` instead of `MfluxInferenceClient`
3. The [API contract](shared/API_CONTRACT.md) stays the same

Cloud options: **[Replicate](https://replicate.com/)** | **[Fal.ai](https://fal.ai/)** | **[RunPod](https://runpod.io/)**

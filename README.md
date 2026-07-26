# mflux Playground

[![CI](https://github.com/stoyanov-x/mflux-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/stoyanov-x/mflux-playground/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](inference-service/pyproject.toml)

> Flux image generation on Apple Silicon — React SPA → ASP.NET Core API → FastAPI/mflux inference.

📖 [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [API contract](shared/API_CONTRACT.md)

---

## Quick Start

**Prerequisites:** Python 3.12+ with [`uv`](https://docs.astral.sh/uv/), Apple Silicon Mac, Docker Desktop (optional).

```bash
./start.sh
```

Installs deps, starts inference natively, Docker containers for web API + frontend, opens http://localhost:5173.

> **First run** downloads the FLUX.2 Klein 4B model (~6 GB). Cached afterwards.  
> **Docker limitation:** MLX needs the Metal GPU — Docker for Mac can't access it, so inference always runs natively.

### Manual start (3 terminals)

| Terminal | Run |
|---|---|
| Inference | `cd inference-service && uv run uvicorn app.main:app --reload --port 8000` |
| Web API + frontend | `docker compose up web-api frontend` |
| Frontend (alt) | `cd web-frontend && npm install && npm run dev` |

---

## What you can do

Three tabs at http://localhost:5173:

| Tab | What it does |
|---|---|
| **Generate** | Prompt + negative prompt, select trained LoRAs (per-adapter scale), upload reference image for img2img, adjust steps/guidance |
| **Train LoRA** | Upload 10–30 images, name your LoRA, configure epochs/steps. Takes 20–60 min on M3 Max. Trained weights appear in Generate tab automatically |
| **History** | Past generations with status badges and image previews |

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  React SPA   │────▶│  ASP.NET Core    │────▶│  FastAPI Inference      │
│  (Vite+TW)   │     │  Web API         │     │  Service (mflux)        │
│  :5173       │◀────│  :5100           │◀────│  :8000                  │
└──────────────┘     └──────────────────┘     │                         │
                          │                   │  ├ Generation worker    │
                     ┌────┴────┐              │  ├ Training runner      │
                     │ SQLite  │              │  ├ MLX cache limit 16GB │
                     │ (EF C)  │              │  ├ os.nice(10)          │
                     └─────────┘              │  └ LoRA store (files)   │
                                              └─────────────────────────┘
```

**Inference service** (`inference-service/`) — FastAPI + mflux (MLX-native). Single-threaded worker, queues jobs. LoRA stacking, img2img, quantisation.  
**Web API** (`web-api/`) — ASP.NET Core 9. In-memory job tracker. `IInferenceClient` abstraction lets you swap the backend.  
**Frontend** (`web-frontend/`) — React 19 + Vite + TypeScript + Tailwind.

---

## Configuration

| Variable | Default (Mac) | Default (Linux) | What |
|---|---|---|---|
| `INFERENCE_API_URL` | `http://host.docker.internal:8000` | `http://inference-service:8000` | Frontend → inference |
| `WEB_API_URL` | `http://web-api:8080` | `http://web-api:8080` | Frontend → web API |

**Resource limits** (env vars, prefix `MFLUX_`): `MLX_CACHE_LIMIT_GB` (16), `LOW_RAM` (true), `NICE_LEVEL` (10).

---

## Supported Models

| Model | Alias | Training | Notes |
|---|---|---|---|
| **FLUX.2 Klein 4B** | `flux2-klein-4b` | ✅ | **Default.** Fastest, smallest, very good |
| FLUX.2 Klein 9B | `flux2-klein-9b` | ✅ | Higher quality, larger |
| Z-Image Turbo | `z-image-turbo` | ✅ | Fast, great realism |
| Z-Image | `z-image` | ✅ | Base Z-Image |
| FLUX.1 dev / schnell | `dev` / `schnell` | ❌ | Legacy, removed from cache |

---

## Development

```bash
make setup-dev    # one-time: install deps + pre-commit hooks
make check        # ruff, pyright, tsc, dotnet build, shellcheck
make trufflehog   # secret scan the full repo
```

Pre-commit hooks on `git commit` (lint, format). Pre-push also runs TruffleHog. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Cloud Portability

The web API's `IInferenceClient` is the swap point — implement it against **[Replicate](https://replicate.com/)** · **[Fal.ai](https://fal.ai/)** · **[RunPod](https://runpod.io/)** without touching the frontend or API contract.

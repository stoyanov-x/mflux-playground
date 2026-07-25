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
- Simple single-page app: submit prompts, adjust LoRA scale / step count, poll for results
- Proxies API calls to the web API during dev

---

## Quick Start (Local Dev)

### Prerequisites
- Python 3.12+ with [`uv`](https://docs.astral.sh/uv/) (or pip)
- .NET 9 SDK
- Node.js 22+
- Apple Silicon Mac (for mflux inference)

### Terminal 1 — Inference Service
```bash
cd inference-service

# Install deps
uv sync

# Start the FastAPI server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first startup downloads the Flux model weights. This is a one-time ~25 GB download.

### Terminal 2 — Web API
```bash
cd web-api
dotnet run --urls "http://localhost:5100"
```

### Terminal 3 — Frontend
```bash
cd web-frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Training LoRAs

Since kohya-ss and ai-toolkit require CUDA, train on ephemeral cloud GPUs:

1. **Spin up a GPU instance** on RunPod / Vast.ai / Lambda (4090 or A100)
2. Install ai-toolkit or kohya-ss
3. Train your style LoRA (typically a few hours, a few dollars)
4. Download the resulting `.safetensors` file
5. Place it in `inference-service/lora_weights/`
6. Reference it in requests: `"lora_paths": ["lora_weights/my_style.safetensors"]`

The LoRA weights are portable — same `.safetensors` file works with mflux, diffusers, and ComfyUI.

---

## Cloud Inference (When You Need to Scale)

The `IInferenceClient` interface in the web API is the swap point.
To move inference to the cloud:

1. Create a new class implementing `IInferenceClient` (e.g., `ReplicateInferenceClient`)
2. Register it in `Program.cs` instead of `MfluxInferenceClient`
3. The inference service contract (`shared/API_CONTRACT.md`) stays the same

Cloud options:
- **[Replicate](https://replicate.com/)** — deploy Flux+LoRA as a serverless endpoint
- **[Fal.ai](https://fal.ai/)** — similar, with Flux-optimised runtimes
- **[RunPod](https://runpod.io/)** — serverless endpoints with custom diffusers pipelines

---

## Project Structure

```
mflux-experiment/
├── inference-service/          # FastAPI + mflux (Apple Silicon)
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── config.py           # pydantic-settings
│   │   ├── models.py           # Pydantic schemas
│   │   ├── inference.py        # mflux wrapper (the swappable bit)
│   │   ├── routers/
│   │   │   └── generate.py     # /api/v1/generate, /jobs/{id}
│   │   └── services/
│   │       ├── job_store.py    # In-memory job storage
│   │       └── job_worker.py   # Background generation worker
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
├── web-api/                    # ASP.NET Core 9
│   ├── Controllers/
│   │   └── GenerationController.cs
│   ├── Models/
│   │   └── GenerationRequest.cs
│   ├── Services/
│   │   ├── IInferenceClient.cs         # Abstraction
│   │   ├── MfluxInferenceClient.cs     # Local impl
│   │   └── GenerationJobService.cs     # In-memory tracker
│   ├── Program.cs
│   ├── web-api.csproj
│   └── Dockerfile
├── web-frontend/               # React 19 + Vite + TypeScript
│   ├── src/
│   │   └── App.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
├── shared/
│   └── API_CONTRACT.md         # Inference service contract
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **mflux over ComfyUI+MPS** | 5-10× faster on Apple Silicon; native MLX, not PyTorch-via-MPS |
| **Async job queue** | M3 Max handles one generation at a time; queue prevents overload and keeps the API responsive |
| **`IInferenceClient` abstraction** | Swap local Mac inference to cloud without touching the web API logic |
| **In-memory stores** | Good enough for a personal project; swap for Redis/Postgres when scaling |
| **LoRA weights as files** | Portable across mflux, diffusers, ComfyUI — no training lock-in |

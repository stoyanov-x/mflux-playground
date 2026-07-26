# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2025-07-26

### Added

- **Three-tier architecture**: React SPA → ASP.NET Core Web API → FastAPI inference service
- **Image generation**: Prompt + negative prompt, LoRA adapter stacking (multi, with per-adapter scale), img2img (reference image + denoise strength), configurable steps/guidance/seed
- **LoRA training**: Upload 10–30 images via drag & drop, train with configurable epochs/steps, trained `.safetensors` appears automatically in the generation LoRA selector
- **Job history**: Track past generations with status badges, progress, and image previews
- **Model support**: FLUX.2 Klein 4B (default), FLUX.2 Klein 9B, Z-Image Turbo, Z-Image
- **Cloud-portable contract**: `IInferenceClient` abstraction in the Web API — swap the backend to Replicate, Fal.ai, or RunPod without touching the frontend
- **Hybrid local dev**: Inference runs natively (MLX needs Metal GPU on Apple Silicon), web API and frontend in Docker
- **CI pipeline**: Python lint/typecheck, frontend TS build, .NET build+format, shellcheck, Docker build smoke test, TruffleHog secret scanning
- **Pre-commit hooks**: trailing-whitespace, EOF fixer, YAML check, shellcheck, ruff lint+format, TruffleHog (pre-push)
- **Quality tooling**: Ruff (lint+format), Pyright (type checking), shellcheck, `make check` to run all gates locally

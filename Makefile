.PHONY: dev inference api frontend hybrid clean setup setup-dev lint format check ruff pyright

# ── Quick-start ──────────────────────────────────────────────
#   Three-terminals (recommended for Mac local dev):
#     Terminal 1: make inference      (native — requires Apple Silicon + mflux)
#     Terminal 2: make docker-svc     (web-api + frontend in Docker)
#     Terminal 3: (n/a — docker-svc handles both)
#
#   Or the hybrid shortcut:
#     Terminal 1: make inference
#     Terminal 2: make hybrid          (starts web-api + frontend)

dev:
	@echo "══════════════════════════════════════════════════════"
	@echo "  Local dev on Mac (recommended)"
	@echo "══════════════════════════════════════════════════════"
	@echo ""
	@echo "  Terminal 1 (native — inference requires MLX/Metal):"
	@echo "    make inference"
	@echo ""
	@echo "  Terminal 2 (Docker — web API + frontend):"
	@echo "    make hybrid"
	@echo ""
	@echo "  Or all native (three terminals):"
	@echo "    make inference   — FastAPI inference (mflux)"
	@echo "    make api         — ASP.NET Core web API"
	@echo "    make frontend    — Vite + React dev server"
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo "  Cloud / Linux (all in Docker, NVIDIA GPU required):"
	@echo "    docker compose up --build"
	@echo "══════════════════════════════════════════════════════"

## Start the mflux inference service natively (requires Apple Silicon + MLX)
inference:
	cd inference-service && \
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## Hybrid: web-api + frontend in Docker, inference native
hybrid:
	docker compose up --build web-api frontend

## Start the ASP.NET Core web API natively
api:
	cd web-api && dotnet run --urls "http://localhost:5100"

## Start the Vite + React frontend natively
frontend:
	cd web-frontend && npm run dev

## Install all dependencies
setup:
	cd inference-service && uv sync
	cd web-frontend && npm install

## Setup dev environment (deps + pre-commit hooks)
setup-dev: setup
	brew list pre-commit &>/dev/null || brew install pre-commit
	pre-commit install --hook-type pre-commit --hook-type pre-push
	@echo "✅ pre-commit hooks installed"

## Run TruffleHog secret scanning on the full repo
trufflehog:
	@echo "🔍 Installing TruffleHog if missing…"
	@which trufflehog >/dev/null 2>&1 || brew install trufflehog
	@echo "🔍 Scanning $(PWD) for secrets…"
	trufflehog filesystem --only-verified --fail --no-update "$(PWD)"

## Lint all shell scripts with shellcheck
lint:
	shellcheck start.sh shutdown.sh

## Python lint (ruff)
ruff:
	cd inference-service && uv run ruff check

## Python type check (pyright)
pyright:
	cd inference-service && uv run pyright

## Python format check (ruff format)
format-py:
	cd inference-service && uv run ruff format

## C# format check
format-cs:
	cd web-api && dotnet format --verify-no-changes

## Run all quality checks locally
check: lint ruff pyright
	cd web-frontend && npx tsc -b
	cd web-api && dotnet build --configuration Release
	@echo ""
	@echo "✅ All checks passed"

## Clean generated artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name node_modules -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name bin -path '*/web-api/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name obj -path '*/web-api/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf web-frontend/dist

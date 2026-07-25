.PHONY: dev inference api frontend clean setup

# ── Quick-start (three terminals) ────────────────────────────
#   Terminal 1: make inference
#   Terminal 2: make api
#   Terminal 3: make frontend

dev:
	@echo "Open three terminals and run:"
	@echo "  make inference   — FastAPI inference service (mflux)"
	@echo "  make api         — ASP.NET Core web API"
	@echo "  make frontend    — Vite + React dev server"

## Start the mflux inference service (requires Python + MLX)
inference:
	cd inference-service && \
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## Start the ASP.NET Core web API
api:
	cd web-api && dotnet run --urls "http://localhost:5100"

## Start the Vite + React frontend
frontend:
	cd web-frontend && npm run dev

## Install all dependencies
setup:
	cd inference-service && uv sync
	cd web-frontend && npm install

## Clean generated artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name node_modules -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name bin -path '*/web-api/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name obj -path '*/web-api/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf web-frontend/dist

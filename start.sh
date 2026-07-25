#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  mflux Experiment — hybrid local dev launcher
# ══════════════════════════════════════════════════════════════
#  Starts everything with one command:
#    • Inference service  ── natively (MLX needs Metal GPU)
#    • Web API            ── in Docker
#    • Frontend           ── in Docker
#    • Browser            ── opened automatically
#
#  Usage:  ./start.sh
#  Press Ctrl+C to stop everything.
# ══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }

# ── Prerequisite checks ─────────────────────────────────────

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        err "$1 is required but not found. Install it first."
        case "$1" in
            uv)    echo "       → curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
            docker) echo "       → https://docs.docker.com/desktop/install/mac-install/" ;;
            python3) echo "       → brew install python@3.12" ;;
        esac
        exit 1
    fi
}

info "Checking prerequisites…"
check_cmd uv
check_cmd docker
check_cmd python3
ok "All prerequisites found."

# ── Detect Apple Silicon ────────────────────────────────────

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    warn "Not running on Apple Silicon (arm64)."
    warn "mflux requires MLX which only runs on Apple Silicon."
    warn "The inference service will likely fail."
    echo ""
fi

# ── Install Python dependencies ─────────────────────────────

info "Installing inference service dependencies…"
cd "$SCRIPT_DIR/inference-service"
uv venv 2>/dev/null || true   # create .venv if not present
uv pip install -r requirements.txt
ok "Python dependencies installed."

cd "$SCRIPT_DIR"

# ── Check if Docker is running ──────────────────────────────

if ! docker info &>/dev/null; then
    err "Docker is not running. Start Docker Desktop first."
    exit 1
fi
ok "Docker is running."

# ── Cleanup handler ─────────────────────────────────────────

cleanup() {
    echo ""
    info "Shutting down…"
    if [[ -n "${INFERENCE_PID:-}" ]]; then
        kill "$INFERENCE_PID" 2>/dev/null || true
        wait "$INFERENCE_PID" 2>/dev/null || true
        ok "Inference service stopped."
    fi
    info "Stopping Docker services…"
    docker compose down 2>/dev/null || true
    ok "Docker services stopped."
    info "Bye!"
}

trap cleanup EXIT INT TERM

# ── Start Docker services ───────────────────────────────────

info "Starting web API and frontend in Docker…"
docker compose up --build -d web-api frontend
ok "Docker services started."

# ── Wait for Docker health ──────────────────────────────────

wait_for_port() {
    local host="$1" port="$2" service="$3" max_attempts="${4:-30}"
    local attempt=0
    while ! nc -z "$host" "$port" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            err "$service not ready after $max_attempts seconds."
            err "Check 'docker compose logs $service' for details."
            return 1
        fi
        sleep 1
    done
    ok "$service is ready on $host:$port"
}

wait_for_port "localhost" 5100 "Web API" 60
wait_for_port "localhost" 5173 "Frontend" 30

# ── Start inference service natively ────────────────────────

info "Starting inference service (native — MLX + Metal)…"
cd "$SCRIPT_DIR/inference-service"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info &
INFERENCE_PID=$!
cd "$SCRIPT_DIR"

# Wait for inference to be ready (may need to download model on first run)
info "Waiting for inference service to be ready (may download ~6GB model on first run)…"
local attempt=0
while true; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        ok "Inference service is ready."
        break
    fi
    attempt=$((attempt + 1))
    if [[ $attempt -ge 300 ]]; then  # 5 minute timeout
        warn "Inference service not ready after 5 minutes."
        warn "Check logs with: docker compose logs inference-service"
        break
    fi
    sleep 1
done

# ── Print summary ───────────────────────────────────────────

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Everything is running!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Frontend:${NC}          http://localhost:5173"
echo -e "  ${CYAN}Web API:${NC}           http://localhost:5100"
echo -e "  ${CYAN}Inference API:${NC}     http://localhost:8000"
echo -e "  ${CYAN}Inference docs:${NC}    http://localhost:8000/docs"
echo ""
echo -e "  ${YELLOW}Press Ctrl+C to stop everything.${NC}"
echo ""

# Open browser
if command -v open &>/dev/null; then
    open http://localhost:5173
elif command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5173
fi

# ── Wait for Ctrl+C ─────────────────────────────────────────
wait

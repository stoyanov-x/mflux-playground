#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  mflux Experiment — shutdown
# ══════════════════════════════════════════════════════════════
#  Stops everything started by start.sh (or any manual combo):
#    • Inference service  ── kills any uvicorn/mflux processes
#    • Web API + Frontend ── docker compose down
#
#  Safe to run even if nothing is running.
#  Usage:  ./shutdown.sh
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

# ── Kill inference service ──────────────────────────────────

info "Stopping inference service…"

# Kill uvicorn processes started from the inference-service directory
PIDS=$(pgrep -f "uvicorn app.main:app" 2>/dev/null || true)
if [[ -n "$PIDS" ]]; then
    echo "$PIDS" | while read -r pid; do
        kill "$pid" 2>/dev/null && ok "Killed inference process (PID $pid)" || warn "Could not kill PID $pid (already gone?)"
    done
else
    warn "No inference service process found."
fi

# Also catch any lingering mflux processes (orphaned by a crashed terminal)
MFLUX_PIDS=$(pgrep -f "mflux" 2>/dev/null || true)
if [[ -n "$MFLUX_PIDS" ]]; then
    while read -r pid; do
        # Only kill mflux processes that aren't part of this script
        if [[ "$pid" != "$$" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done <<< "$MFLUX_PIDS"
fi

# ── Stop Docker services ────────────────────────────────────

info "Stopping Docker services…"
if docker compose ls 2>/dev/null | grep -q .; then
    docker compose down 2>/dev/null || true
    ok "Docker services stopped."
else
    warn "No active Docker Compose project found."
fi

# ── Clean up .venv prompt ───────────────────────────────────
# Ask if the user wants to remove the Python virtual environment
# (skip in non-interactive mode)
if [[ -t 0 ]]; then
    echo ""
    info "The inference service Python virtual environment is at:"
    echo "    $SCRIPT_DIR/inference-service/.venv"
    echo "  (contains ~25 GB of cached model weights)"
    echo ""
    read -r -p "Remove .venv and cached model weights? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            info "Removing .venv…"
            rm -rf "$SCRIPT_DIR/inference-service/.venv"
            ok "Virtual environment removed."
            ;;
        *)
            info "Keeping .venv (faster next startup)."
            ;;
    esac
fi

# ── Summary ──────────────────────────────────────────────────

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Shutdown complete.${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""

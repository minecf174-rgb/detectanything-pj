#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != "--check" ]]; then
    fail "Usage: bash setup-scripts/setup-new-machine.sh [--check]"
fi

echo "========================================="
echo " Detect Anything - New Machine Setup"
echo "========================================="

[[ -f "$ROOT_DIR/pyproject.toml" && -f "$ROOT_DIR/alembic.ini" && -f "$ROOT_DIR/frontend/package-lock.json" ]] \
    || fail "Project files are missing. Run this script from a complete project checkout."

command -v git >/dev/null 2>&1 || fail "Git is required to install the Python dependencies."
command -v node >/dev/null 2>&1 || fail "Node.js 20+ is required. Install it, then run this script again."
command -v npm >/dev/null 2>&1 || fail "npm is required. Install Node.js with npm, then run this script again."
node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' \
    || fail "Node.js 20+ is required (found $(node --version))."

# The platform installers reuse an active virtual environment. Do not let an
# unrelated venv receive this project's packages by accident.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    active_venv="$(cd "$VIRTUAL_ENV" && pwd -P)"
    [[ "$active_venv" == "$ROOT_DIR/venv" ]] \
        || fail "An unrelated virtual environment is active: $VIRTUAL_ENV. Deactivate it and run this script again."
fi

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
    python_for_check="$ROOT_DIR/venv/bin/python"
else
    python_for_check=""
    for candidate in python3.12 python3.11; do
        if command -v "$candidate" >/dev/null 2>&1; then
            python_for_check="$(command -v "$candidate")"
            break
        fi
    done
    [[ -n "$python_for_check" ]] \
        || fail "Python 3.11 or 3.12 is required. Install it, then run this script again."
fi

"$python_for_check" -c 'import sys; sys.exit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else 1)' \
    || fail "The project venv must use Python 3.11 or 3.12 (found $("$python_for_check" --version))."

case "$(uname -s)" in
    Darwin)
        platform_script="$ROOT_DIR/setup-scripts/setup-mac.sh"
        platform_name="macOS (CPU/MPS)"
        ;;
    Linux)
        gpu_list=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            gpu_list="$(nvidia-smi -L 2>/dev/null || true)"
        fi
        if [[ "$gpu_list" == *"GPU 0:"* ]]; then
            if [[ -n "${PYTORCH_CUDA:-}" ]]; then
                case "$PYTORCH_CUDA" in
                    cu118|cu124|cu126) cuda_build="$PYTORCH_CUDA" ;;
                    *) fail "PYTORCH_CUDA must be cu118, cu124, or cu126 for PyTorch 2.6.0." ;;
                esac
            else
                # nvidia-smi reports the maximum CUDA version supported by the driver.
                driver_cuda="$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/p' || true)"
                if [[ "$driver_cuda" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
                    cuda_major="${BASH_REMATCH[1]}"
                    cuda_minor="${BASH_REMATCH[2]}"
                    if (( cuda_major > 12 || (cuda_major == 12 && cuda_minor >= 6) )); then
                        cuda_build="cu126"
                    elif (( cuda_major == 12 && cuda_minor >= 4 )); then
                        cuda_build="cu124"
                    elif (( cuda_major == 12 || (cuda_major == 11 && cuda_minor >= 8) )); then
                        cuda_build="cu118"
                    else
                        cuda_build=""
                    fi
                else
                    cuda_build=""
                fi
            fi

            if [[ -n "$cuda_build" ]]; then
                platform_script="$ROOT_DIR/setup-scripts/setup-linux-cuda.sh"
                platform_name="Linux NVIDIA CUDA ($cuda_build)"
            else
                echo "WARNING: NVIDIA GPU found, but its driver CUDA version is unavailable or below 11.8."
                echo "         Installing CPU-only PyTorch. Update the driver to enable GPU support."
                platform_script="$ROOT_DIR/setup-scripts/setup-linux-cpu.sh"
                platform_name="Linux CPU"
            fi
        else
            platform_script="$ROOT_DIR/setup-scripts/setup-linux-cpu.sh"
            platform_name="Linux CPU"
        fi
        ;;
    *)
        fail "This .sh script supports macOS and Linux only (including Linux with NVIDIA CUDA)."
        ;;
esac

echo "Platform: $platform_name"
echo "Python: $("$python_for_check" --version)"
echo "Node.js: $(node --version)"
echo "npm: $(npm --version)"
echo

if [[ "${1:-}" == "--check" ]]; then
    echo "Check complete; no packages installed and no database changes made."
    exit 0
fi

if [[ "$platform_script" == *setup-linux-cuda.sh ]]; then
    PYTORCH_CUDA="$cuda_build" bash "$platform_script"
else
    bash "$platform_script"
fi

PYTHON="$ROOT_DIR/venv/bin/python"
[[ -x "$PYTHON" ]] || fail "Project virtual environment was not created at $ROOT_DIR/venv."

echo
echo "Installing frontend packages from package-lock.json..."
(
    cd "$ROOT_DIR/frontend"
    npm ci
)

# Preserve existing SQLite data before Alembic changes its schema. The backup
# API also includes committed data in a SQLite WAL file if one is present.
"$PYTHON" - <<'PY'
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
import sqlite3

from sqlalchemy.engine import make_url
from app.core.config.settings import get_settings

url = make_url(get_settings().database_url)
if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
    database = Path(url.database).expanduser()
    if database.is_file():
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = backup_dir / f"{database.name}.pre-setup-{stamp}-{uuid4().hex[:8]}.bak"
        with sqlite3.connect(database) as source, sqlite3.connect(backup) as target:
            source.backup(target)
        print(f"Existing SQLite database backed up to: {backup}")
else:
    print("Non-file SQLite database: automatic backup skipped; back it up separately if needed.")
PY

echo
echo "Applying database migrations..."
"$PYTHON" -m alembic upgrade head
"$PYTHON" -m alembic current

echo
echo "========================================="
echo " New machine setup completed"
echo "========================================="
echo "Backend:  source venv/bin/activate && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
echo "Frontend: cd frontend && npm run dev"
echo "For public access, configure AUTH_JWT_SECRET and tunnel settings in the root .env first."

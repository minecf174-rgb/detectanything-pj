#!/usr/bin/env bash

set -euo pipefail

TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "========================================="
echo " Detect Anything - Linux CPU Setup"
echo "========================================="

# --------------------------------------------------
# OS Check
# --------------------------------------------------

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: This script is for Linux only."
    exit 1
fi

# --------------------------------------------------
# Python / Virtual Environment
# --------------------------------------------------

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active virtual environment:"
    echo "$VIRTUAL_ENV"

    PYTHON="$VIRTUAL_ENV/bin/python"

elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
    echo "Using existing venv/"
    PYTHON="$ROOT_DIR/venv/bin/python"

else
    echo "No venv found. Creating venv..."

    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
    elif command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="python3.11"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "ERROR: Python 3 not found."
        exit 1
    fi

    "$PYTHON_BIN" -m venv venv
    PYTHON="$ROOT_DIR/venv/bin/python"
fi

echo
echo "Python:"
"$PYTHON" --version

# --------------------------------------------------
# Upgrade pip tooling
# --------------------------------------------------

echo
echo "Upgrading pip / setuptools / wheel..."

"$PYTHON" -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# --------------------------------------------------
# PyTorch CPU ONLY
#
# Important:
# Use PyTorch CPU index so pip does NOT download
# NVIDIA / CUDA / cuDNN / NCCL packages.
# --------------------------------------------------

echo
echo "Installing CPU-only PyTorch..."

"$PYTHON" -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    --index-url https://download.pytorch.org/whl/cpu

# --------------------------------------------------
# Verify CPU build before continuing
# --------------------------------------------------

echo
echo "Checking PyTorch build..."

"$PYTHON" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.is_available())

if torch.version.cuda is not None:
    raise SystemExit(
        "ERROR: CUDA-enabled PyTorch detected. "
        "CPU-only PyTorch was expected."
    )
PY

# --------------------------------------------------
# Install project
#
# Disable SAM2 CUDA build.
#
# --no-build-isolation is important here because
# otherwise SAM2 may create another environment and
# download CUDA PyTorch dependencies again.
# --------------------------------------------------

echo
echo "Installing project dependencies..."

SAM2_BUILD_CUDA=0 \
"$PYTHON" -m pip install \
    --no-build-isolation \
    -e ".[dev]"

# --------------------------------------------------
# Dependency check
# --------------------------------------------------

echo
echo "Checking dependencies..."

"$PYTHON" -m pip check

# --------------------------------------------------
# Verify installation
# --------------------------------------------------

echo
echo "========================================="
echo " Verification"
echo "========================================="

"$PYTHON" - <<'PY'
import platform
import torch
import torchvision
import cv2
import numpy
import fastapi
import sqlalchemy
import ultralytics
from ultralytics.nn.tasks import _SafeLoad
import sam2
import segment_anything

print("OS:", platform.platform())
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("FastAPI:", fastapi.__version__)
print("SQLAlchemy:", sqlalchemy.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Safe model loading:", _SafeLoad.SUPPORTED)
print("SAM2: OK")
print("SAM1: OK")

if not _SafeLoad.SUPPORTED:
    raise SystemExit("ERROR: Safe YOLO model loading is unavailable. PyTorch 2.6+ is required.")
PY

echo
echo "========================================="
echo " Linux CPU setup completed"
echo "========================================="
echo
echo "Activate environment with:"
echo "source venv/bin/activate"

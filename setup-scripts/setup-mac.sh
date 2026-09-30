#!/usr/bin/env bash

set -euo pipefail

TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "========================================="
echo " Detect Anything - macOS Setup"
echo "========================================="

# --------------------------------------------------
# OS Check
# --------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: This script is for macOS only."
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
# PyTorch for macOS
# --------------------------------------------------

echo
echo "Installing PyTorch for macOS..."

"$PYTHON" -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION"

# --------------------------------------------------
# Install project
#
# Disable SAM2 CUDA extension:
# macOS does not use NVIDIA CUDA.
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
import sam2
import segment_anything

print("OS:", platform.platform())
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("FastAPI:", fastapi.__version__)
print("SQLAlchemy:", sqlalchemy.__version__)
print("Ultralytics:", ultralytics.__version__)
print("SAM2: OK")
print("SAM1: OK")
PY

echo
echo "========================================="
echo " macOS setup completed"
echo "========================================="
echo
echo "Activate environment with:"
echo "source venv/bin/activate"
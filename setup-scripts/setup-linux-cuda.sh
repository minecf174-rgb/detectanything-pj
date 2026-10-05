#!/usr/bin/env bash

set -euo pipefail

TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"

# Default PyTorch CUDA build.
#
# Can override when running:
#
# PYTORCH_CUDA=cu126 bash setup-scripts/setup-linux-cuda.sh
#
PYTORCH_CUDA="${PYTORCH_CUDA:-cu124}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "========================================="
echo " Detect Anything - Linux CUDA Setup"
echo "========================================="

# --------------------------------------------------
# OS Check
# --------------------------------------------------

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: This script is for Linux only."
    exit 1
fi

# --------------------------------------------------
# NVIDIA Check
# --------------------------------------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found."
    echo
    echo "NVIDIA driver may not be installed correctly."
    echo "Use setup-linux-cpu.sh if this machine has no NVIDIA GPU."
    exit 1
fi

echo
echo "NVIDIA GPU:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# --------------------------------------------------
# Validate PyTorch CUDA build
# --------------------------------------------------

case "$PYTORCH_CUDA" in
    cu118|cu124|cu126)
        ;;
    *)
        echo
        echo "ERROR: Unsupported PyTorch CUDA build:"
        echo "$PYTORCH_CUDA"
        echo
        echo "Available for PyTorch 2.6.0:"
        echo "  cu118"
        echo "  cu124"
        echo "  cu126"
        exit 1
        ;;
esac

echo
echo "PyTorch CUDA build: $PYTORCH_CUDA"

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
# Install CUDA-enabled PyTorch
# --------------------------------------------------

echo
echo "Installing CUDA-enabled PyTorch ($PYTORCH_CUDA)..."

"$PYTHON" -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA"

# --------------------------------------------------
# Verify CUDA before continuing
# --------------------------------------------------

echo
echo "Checking CUDA..."

"$PYTHON" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "\nERROR: PyTorch was installed with CUDA support "
        "but CUDA is not available.\n"
        "Check NVIDIA driver / GPU compatibility."
    )

print("GPU:", torch.cuda.get_device_name(0))
PY

# --------------------------------------------------
# Install project
#
# SAM2 will try to build its CUDA extension.
#
# BUILD_ALLOW_ERRORS=1 means SAM2 installation
# can continue if its optional CUDA extension
# cannot be compiled (for example nvcc is missing).
# --------------------------------------------------

echo
echo "Installing project dependencies..."

SAM2_BUILD_CUDA=1 \
SAM2_BUILD_ALLOW_ERRORS=1 \
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
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )

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
echo " Linux CUDA setup completed"
echo "========================================="
echo
echo "Activate environment with:"
echo "source venv/bin/activate"

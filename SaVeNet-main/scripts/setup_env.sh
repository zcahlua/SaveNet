#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="savenet-mol3d"
PYTHON_VERSION="3.10"
TORCH_VERSION="2.4.0"
CUDA_TAG="cpu"

usage() {
  cat <<EOF
Usage:
  bash scripts/setup_env.sh [--env ENV_NAME] [--python PYTHON_VERSION] [--torch TORCH_VERSION] [--cuda CUDA_TAG]

Examples:
  CPU:
    bash scripts/setup_env.sh --cuda cpu

  CUDA 11.8:
    bash scripts/setup_env.sh --torch 2.4.0 --cuda cu118

  CUDA 12.1:
    bash scripts/setup_env.sh --torch 2.4.0 --cuda cu121

Notes:
  CUDA_TAG should match PyG wheel tags, e.g. cpu, cu118, cu121, cu124.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --torch)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --cuda)
      CUDA_TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not installed or not on PATH. Install Miniconda/Anaconda first."
  exit 1
fi

echo "[1/6] Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip

# Enable `conda activate` in non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[2/6] Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo "[3/6] Installing PyTorch ${TORCH_VERSION} for ${CUDA_TAG}"
if [[ "${CUDA_TAG}" == "cpu" ]]; then
  python -m pip install \
    torch=="${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install \
    torch=="${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

echo "[4/6] Installing PyG packages"
python -m pip install torch_geometric

# SaVeNet currently imports torch_scatter and torch_cluster directly.
python -m pip install \
  torch_scatter \
  torch_cluster \
  torch_sparse \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "[5/6] Installing Molecule3D / MoleculeX dependencies"
python -m pip install -r "${PROJECT_DIR}/requirements-molecule3d.txt"

echo "[6/6] Checking environment"
python "${PROJECT_DIR}/scripts/check_env.py"

echo ""
echo "Environment setup complete."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python -m pip install --upgrade pip

echo "[1/3] Checking PyTorch..."
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception as e:
    raise SystemExit(
        "PyTorch is not installed. Run: bash scripts/setup_env.sh --cuda cpu"
    ) from e
PY

echo "[2/3] Checking PyG extensions..."
python - <<'PY'
missing = []
for name in ["torch_geometric", "torch_scatter", "torch_cluster", "torch_sparse"]:
    try:
        __import__(name)
        print("ok:", name)
    except Exception:
        missing.append(name)

if missing:
    raise SystemExit(
        "Missing PyG packages: "
        + ", ".join(missing)
        + "\nRun scripts/setup_env.sh, or manually install PyG packages matching your torch/CUDA version."
    )
PY

echo "[3/3] Installing MoleculeX / molx dependencies..."
python -m pip install -r "${PROJECT_DIR}/requirements-molecule3d.txt"

python "${PROJECT_DIR}/scripts/check_env.py"

echo "Done."

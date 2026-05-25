#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

echo "[1/3] Checking PyTorch..."
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception as e:
    raise SystemExit(
        "PyTorch is not installed. Install torch first for your CUDA/CPU environment."
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
        + "\nInstall torch-geometric / torch-scatter / torch-cluster / torch-sparse "
          "with wheels matching your installed torch/CUDA version."
    )
PY

echo "[3/3] Installing MoleculeX / molx dependencies..."
python -m pip install -r requirements-molecule3d.txt

python - <<'PY'
from molx.dataset import Molecule3DProps
print("MoleculeX import OK:", Molecule3DProps)
PY

echo "Done."

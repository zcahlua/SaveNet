# SaVeNet Reference Implementation

SaVeNet here is used as a **3D molecular property prediction model** using Molecule3D ground-truth 3D coordinates (`xyz`/`pos`). This pipeline does **not** perform 2D-to-3D generation.

## Environment setup
Install Python packages compatible with your local PyTorch/CUDA setup:
- torch
- torch-geometric
- torch-scatter
- torch-cluster
- torch-sparse
- rdkit
- pandas
- numpy
- tqdm
- moleculex / molx (for `Molecule3DProps`)
- ase

> Note: PyG extension wheels (`torch-scatter`, `torch-cluster`, `torch-sparse`) must match your torch/CUDA version.

## Dataset layout
Expected layout:

```text
<DATA_ROOT>/
  data/
    raw/
      properties.csv
      random_split_inds.json
      scaffold_split_inds.json
      random_test_split_inds.json
      scaffold_test_split_inds.json
      combined_mols_0_to_1000000.sdf
      combined_mols_1000000_to_2000000.sdf
      combined_mols_2000000_to_3000000.sdf
      combined_mols_3000000_to_3899647.sdf
    processed/ or processed_downstream_random/
```

## Target IDs
- 0: Dipole x
- 1: Dipole y
- 2: Dipole z
- 3: HOMO
- 4: LUMO
- 5: HOMO-LUMO Gap (default)
- 6: SCF Energy

## Tests
```bash
python -m pytest tests -q
```

## Smoke train run
```bash
python train_molecule3d.py \
  --data-root /path/to/Molecule3D \
  --target-id 5 \
  --split-mode random \
  --epochs 1 \
  --batch-size 8 \
  --subset-train 256 \
  --subset-val 64 \
  --subset-test 64 \
  --device cuda
```

## Full training
```bash
python train_molecule3d.py \
  --data-root /path/to/Molecule3D \
  --target-id 5 \
  --split-mode random \
  --epochs 100 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-encoder 8 \
  --num-rbf 32 \
  --cutoff 5.0 \
  --out-dir runs/molecule3d_savenet_gap
```

## Install MoleculeX / molx and prepare Molecule3D

This project uses the MoleculeX dataset class:

```python
from molx.dataset import Molecule3DProps
```

The pip package name is `moleculex`, but the import name is `molx`.

### Step 1: Install PyTorch and PyG

Install `torch`, `torch-geometric`, `torch-scatter`, `torch-cluster`, and `torch-sparse`
using wheels compatible with your CUDA/Torch version.

### Step 2: Install MoleculeX

```bash
cd SaVeNet-main
bash scripts/install_moleculex.sh
```

Or manually:

```bash
pip install moleculex==0.0.3
```

### Step 3: Download Molecule3D raw data

Download the official Molecule3D raw data from the Google Drive link in the MoleculeX README.
Place files here:

```text
/path/to/Molecule3D/
  data/
    raw/
      properties.csv
      random_split_inds.json
      scaffold_split_inds.json
      random_test_split_inds.json
      scaffold_test_split_inds.json
      combined_mols_0_to_1000000.sdf
      combined_mols_1000000_to_2000000.sdf
      combined_mols_2000000_to_3000000.sdf
      combined_mols_3000000_to_3899647.sdf
```

### Step 4: Preprocess Molecule3D

```bash
python scripts/prepare_molecule3d.py \
  --data-root /path/to/Molecule3D \
  --split-mode random \
  --process-dir-base processed_downstream
```

This creates:

```text
/path/to/Molecule3D/data/processed_downstream_random/
  train.pt
  val.pt
  test.pt
```

### Step 5: Train SaVeNet

```bash
python train_molecule3d.py \
  --data-root /path/to/Molecule3D \
  --target-id 5 \
  --split-mode random \
  --epochs 1 \
  --batch-size 4 \
  --subset-train 32 \
  --subset-val 16 \
  --subset-test 16 \
  --device cpu
```

#!/usr/bin/env python
from __future__ import annotations

import importlib
import sys


REQUIRED_IMPORTS = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_cluster",
    "torch_sparse",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
    "ase",
    "rdkit",
    "molx",
]


def check_import(name: str) -> None:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"ok: {name} version={version}")


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))

    failed = []
    for name in REQUIRED_IMPORTS:
        try:
            check_import(name)
        except Exception as exc:
            failed.append((name, exc))

    try:
        from molx.dataset import Molecule3DProps

        print("ok: from molx.dataset import Molecule3DProps")
        print("Molecule3DProps:", Molecule3DProps)
    except Exception as exc:
        failed.append(("molx.dataset.Molecule3DProps", exc))

    try:
        import torch

        print("torch.cuda.is_available:", torch.cuda.is_available())
        print("torch.version.cuda:", torch.version.cuda)
    except Exception:
        pass

    if failed:
        print("\nEnvironment check failed:")
        for name, exc in failed:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()

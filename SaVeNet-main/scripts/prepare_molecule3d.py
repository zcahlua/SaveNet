#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


DATASET_URL = "https://drive.google.com/drive/u/2/folders/1y-EyoDYMvWZwClc2uvXrM4_hQBtM85BI?usp=sharing"

RAW_FILES = [
    "properties.csv",
    "random_split_inds.json",
    "scaffold_split_inds.json",
    "random_test_split_inds.json",
    "scaffold_test_split_inds.json",
    "combined_mols_0_to_1000000.sdf",
    "combined_mols_1000000_to_2000000.sdf",
    "combined_mols_2000000_to_3000000.sdf",
    "combined_mols_3000000_to_3899647.sdf",
]


def check_raw_files(data_root: Path) -> None:
    raw_dir = data_root / "data" / "raw"
    missing = [name for name in RAW_FILES if not (raw_dir / name).exists()]

    if missing:
        msg = [
            f"Missing Molecule3D raw files in: {raw_dir}",
            "",
            "Download the official Molecule3D raw data from:",
            f"  {DATASET_URL}",
            "",
            "Then place the files under:",
            f"  {raw_dir}",
            "",
            "Missing files:",
        ]
        msg.extend(f"  - {name}" for name in missing)
        raise SystemExit("\n".join(msg))

    print(f"Raw file check OK: {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and preprocess Molecule3D for SaVeNet."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to Molecule3D root. Expected layout: <data-root>/data/raw/*.sdf, properties.csv, split JSON files.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "scaffold"],
        default="random",
    )
    parser.add_argument(
        "--process-dir-base",
        default="processed_downstream",
        help="Base processed directory name used by Molecule3DProps.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    check_raw_files(data_root)

    try:
        from molx.dataset import Molecule3DProps
    except ImportError as exc:
        raise SystemExit(
            "Cannot import molx.dataset.Molecule3DProps. "
            "Install MoleculeX first, for example: pip install moleculex==0.0.3"
        ) from exc

    print("Starting Molecule3DProps preprocessing.")
    print(f"data_root: {data_root}")
    print(f"split_mode: {args.split_mode}")
    print(f"process_dir_base: {args.process_dir_base}")

    for split in ["train", "val", "test"]:
        ds = Molecule3DProps(
            root=str(data_root),
            split=split,
            split_mode=args.split_mode,
            process_dir_base=args.process_dir_base,
        )
        print(f"{split}: {len(ds)} samples")

    processed_dir = data_root / "data" / f"{args.process_dir_base}_{args.split_mode}"
    print(f"Processed data should now be under: {processed_dir}")


if __name__ == "__main__":
    main()

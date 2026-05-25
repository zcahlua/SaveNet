"""Utilities for adapting Molecule3D/MoleculeX data to SaVeNet.

SaVeNet is used here as a 3D property prediction model on ground-truth geometries
(data.xyz -> data.pos), not as a 2D-to-3D geometry generation model.
"""

from __future__ import annotations

from typing import Optional, Tuple


class Molecule3DToSaVeNet:
    def __init__(self, target_id: int = 5, center_positions: bool = True):
        self.target_id = target_id
        self.center_positions = center_positions

    def __call__(self, data):
        data = data.clone()

        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos
        elif hasattr(data, "xyz") and data.xyz is not None:
            pos = data.xyz
        else:
            raise ValueError("Molecule3D sample has neither pos nor xyz coordinates.")
        pos = pos.float()
        if self.center_positions:
            pos = pos - pos.mean(dim=0, keepdim=True)
        data.pos = pos

        if hasattr(data, "z") and data.z is not None:
            z = data.z
        elif hasattr(data, "x") and data.x is not None:
            z = data.x[:, 0].long() + 1
        else:
            raise ValueError("Molecule3D sample has neither z nor x from which z can be recovered.")
        data.z = z.long()

        if hasattr(data, "props") and data.props is not None:
            y = data.props[self.target_id]
        elif hasattr(data, "y") and data.y is not None:
            y = data.y
        else:
            raise ValueError("Molecule3D sample has neither y nor props target vector.")

        data.y = y.view(1).float()
        return data


def _subset(dataset, size: Optional[int]):
    if size is None:
        return dataset
    return dataset[:size]


def load_molecule3d_datasets(root: str, target_id: int = 5, split_mode: str = "random", center_positions: bool = False,
                             process_dir_base: str = "processed_downstream", subset_train: Optional[int] = None,
                             subset_val: Optional[int] = None, subset_test: Optional[int] = None) -> Tuple[object, object, object]:
    try:
        from molx.dataset import Molecule3DProps
    except ImportError as exc:
        raise ImportError("Install MoleculeX / molx; direct processed-file loading is not implemented yet.") from exc

    transform = Molecule3DToSaVeNet(target_id=target_id, center_positions=center_positions)
    train_dataset = Molecule3DProps(root=root, split="train", split_mode=split_mode, transform=transform, process_dir_base=process_dir_base)
    val_dataset = Molecule3DProps(root=root, split="val", split_mode=split_mode, transform=transform, process_dir_base=process_dir_base)
    test_dataset = Molecule3DProps(root=root, split="test", split_mode=split_mode, transform=transform, process_dir_base=process_dir_base)

    return _subset(train_dataset, subset_train), _subset(val_dataset, subset_val), _subset(test_dataset, subset_test)

import logging
import math

import torch

from .layers import BesselBasis, GaussianRBF, get_weight_init_by_string, str2act


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_center_of_mass_torch(atomic_numbers, positions):
    from ase.data import atomic_masses

    torch_masses = torch.tensor(
        atomic_masses, device=atomic_numbers.device, dtype=torch.float32
    )
    masses = torch_masses[atomic_numbers]
    return masses[:, None].T @ positions / masses.sum()


def recover(merged, vector_dim):
    vector = torch.reshape(merged[..., -3 * vector_dim:], merged.shape[:-1] + (3, vector_dim))
    scalar = merged[..., : -3 * vector_dim]
    return scalar, vector


def flatten(scalar, vector):
    flat_vector = torch.reshape(vector, vector.shape[:-2] + (vector.shape[-2] * vector.shape[-1],))
    return torch.cat([scalar, flat_vector], -1)


def _init(activation, bias_init, weight_init):
    if type(weight_init) == str:
        weight_init = get_weight_init_by_string(weight_init)
    if type(bias_init) == str:
        bias_init = get_weight_init_by_string(bias_init)
    if type(activation) is str:
        activation = str2act(activation)
    return activation, bias_init, weight_init


def basis(basis_name):
    if basis_name == "BesselBasis":
        return BesselBasis
    if basis_name == "GaussianRBF":
        return GaussianRBF
    raise ValueError(f"Unknown radial basis: {basis_name}")


INV_SQRT_3 = 1 / math.sqrt(3)

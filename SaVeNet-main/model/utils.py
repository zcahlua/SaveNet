import math

import torch
from ase.data import atomic_masses

from model.layers import BesselBasis, GaussianRBF, get_weight_init_by_string, str2act


def get_center_of_mass_torch(atomic_numbers, positions):
    """
    Computes center of mass.
    Args:
        atoms (ase.Atoms): atoms object of molecule
    Returns:
        center of mass
    """
    torch_masses = torch.tensor(
        atomic_masses, device=atomic_numbers.device, dtype=torch.float32
    )
    masses = torch_masses[atomic_numbers]
    return masses[:, None].T @ positions / masses.sum()


def recover(merged, vector_dim):
    vector = torch.reshape(merged[..., -3 * vector_dim:], merged.shape[:-1] + (vector_dim, 3))
    scalar = merged[..., : -3 * vector_dim]
    return scalar, vector


def flatten(scalar, vector):
    flat_vector = torch.reshape(vector, vector.shape[:-2] + (3 * vector.shape[-2],))
    return torch.cat([scalar, flat_vector], -1)


def _init(activation, bias_init, weight_init):
    """
    Initializes weights, biases and the activation function.

    Args:
    activation (str): Name of the activation function to use.
    bias_init (str): Name of the bias initialization method to use.
    weight_init (str): Name of the weight initialization method to use.

    Returns:
    Tuple[Callable, Callable, Callable]: Initialized activation function, bias initializer, and weight initializer.
    """
    if type(weight_init) == str:
        weight_init = get_weight_init_by_string(weight_init)
    if type(bias_init) == str:
        bias_init = get_weight_init_by_string(bias_init)
    if type(activation) is str:
        activation = str2act(activation)
    return activation, bias_init, weight_init


def basis(basis):
    """
    Determines the type of radial basis function to use.

    Args:
    basis (str): Input basis type ("BesselBasis" or "GaussianRBF").

    Returns:
    Callable: Selected basis function.

    Raises:
    ValueError: If an unknown radial basis is provided.
    """
    if basis == "BesselBasis":
        basis = BesselBasis
    elif basis == "GaussianRBF":
        basis = GaussianRBF
    else:
        raise ValueError("Unknown radial basis: {}".format(basis))
    return basis


INV_SQRT_3 = 1 / math.sqrt(3)

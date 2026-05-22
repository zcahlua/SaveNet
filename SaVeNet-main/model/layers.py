""" SaVeNet Layers """

import inspect
import math
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from torch import nn as nn
from torch.nn.init import constant_, xavier_uniform_
from torch_cluster import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from typing import List

zeros_initializer = partial(constant_, val=0.0)


class CosineCutoff(nn.Module):
    """
    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff=5.0, eps=0.1):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("eps", torch.FloatTensor([eps]))

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs + self.eps


@torch.jit.script
def safe_norm(x: Tensor, dim: int = -2, eps: float = 1e-8, keepdim: bool = False):
    return torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=keepdim)) + eps


class NormalizedProperty(nn.Module):
    def __init__(self, mean, stddev):
        super(NormalizedProperty, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        y = input * self.stddev + self.mean
        return y


class SelectProperty(nn.Module):
    def __init__(self, key):
        super(SelectProperty, self).__init__()
        self.key = key

    def forward(self, inputs):
        return inputs[self.key]


# @torch.jit.script
def dist_dir(pos: Tensor, edge_index: List[Tensor]):
    j, i = edge_index

    dist_vec = pos[i] - pos[j]
    distance = safe_norm(dist_vec, dim=-1)

    dist_vec_norm = dist_vec / distance.unsqueeze(-1)

    return distance, dist_vec_norm


def get_geometry(batch, cutoff=5.0):
    atomic_numbers, pos, batch_idx = batch.z, batch.pos, batch.batch

    if "edge_index" not in batch:
        edge_index = radius_graph(pos, r=cutoff, loop=False, batch=batch_idx)
        batch.edge_index = edge_index
    else:
        edge_index = batch.edge_index

    if "dir_ij" not in batch or "rij" not in batch:
        rij, dir_ij = dist_dir(pos, edge_index=edge_index, num_nodes=pos.size(0))
        batch.rij = rij
        batch.dir_ij = dir_ij

    return batch


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    """
    [schnetpack]
    """
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    """
    [schnetpack] Gaussian radial basis functions.
    """

    def __init__(
            self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)

    Directional message passing for molecular graphs. @ ICLR 2020
    """

    def __init__(self, cutoff=5.0, n_rbf=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.register_buffer("norm1", torch.tensor(1.0))

    def forward(self, inputs):
        input_size = len(inputs.shape)
        a = self.freqs[None, :]
        inputs = inputs[..., None]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, self.norm1, inputs)
        y = sinax / norm

        return y


def glorot_orthogonal_wrapper_(tensor, scale=2.0):
    return glorot_orthogonal(tensor, scale=scale)


def get_weight_init_by_string(init_str):
    if init_str == "":
        # Noop
        return lambda x: x
    elif init_str == "zeros":
        return torch.nn.init.zeros_
    elif init_str == "xavier_uniform":
        return torch.nn.init.xavier_uniform_
    elif init_str == "glo_orthogonal":
        return glorot_orthogonal_wrapper_
    else:
        raise ValueError(f"Unknown initialization {init_str}")


class Dense(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            activation=None,
            weight_init=xavier_uniform_,
            bias_init=zeros_initializer,
            norm=None,
            gain=None,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super(Dense, self).__init__(in_features, out_features, bias)
        # Initialize activation function
        if inspect.isclass(activation):
            self.activation = activation()
        self.activation = activation

        if norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        if self.activation:
            y = self.activation(y)
        return y


class MLP(nn.Module):
    def __init__(
            self,
            hidden_dims: List[int],
            n_layers: int = -1,
            bias=True,
            activation=None,
            last_activation=None,
            weight_init=xavier_uniform_,
            bias_init=zeros_initializer,
    ):
        super().__init__()

        DenseMLP = partial(
            Dense, bias=bias, weight_init=weight_init, bias_init=bias_init
        )

        if n_layers > 1:
            assert (
                    len(hidden_dims) == 3
            ), "n_layers and hidden_dims are mutually exclusive"
            dim_in, dim_hid, dim_out = hidden_dims
            hidden_dims = [dim_in] + [dim_hid] * (n_layers - 1) + [dim_out]

        dims = hidden_dims
        n_layers = len(dims)

        self.dense_layers = nn.ModuleList(
            [
                DenseMLP(dims[i], dims[i + 1], activation=activation)
                for i in range(n_layers - 2)
            ]
            + [DenseMLP(dims[-2], dims[-1], activation=last_activation)]
        )

        self.layers = nn.Sequential(*self.dense_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x):
        return self.layers(x)


def normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def scaled_silu(x, scale=0.6):
    return F.silu(x) * scale


def get_activations(optional=False, *args, **kwargs):
    """
    Get all available activation functions.
    Based on https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14

    Args:
        optional:
        *args: positional arguments to be passed to the activation function
        **kwargs: argument dictionary to be passed to the activation function

    Returns:

    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }

    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "swish": torch.nn.SiLU,
            "selu": torch.nn.SELU,
            "scaled_swish": scaled_silu,
        }
    )

    if optional:
        activations[""] = None

    return activations


def dictionary_to_option(options, selected):
    if selected not in options:
        raise ValueError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))} '
        )

    activation = options[selected]
    if inspect.isclass(activation):
        activation = activation()
    return activation


def str2act(input_str, *args, **kwargs):
    if input_str == "":
        return None

    act = get_activations(optional=True, *args, **kwargs)
    out = dictionary_to_option(act, input_str)
    return out

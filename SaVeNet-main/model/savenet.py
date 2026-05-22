from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.nn import Parameter
from torch_scatter import scatter

from . import decoder as dec
from .layers import MLP, Dense, CosineCutoff, safe_norm, get_geometry
from .utils import INV_SQRT_3, _init, flatten, recover, basis, get_logger

log = get_logger(__name__)


class SaVeNet(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            num_encoder: int = 8,
            num_rbf: int = 32,
            cutoff: float = 5.0,
            r_basis: str = "BesselBasis",
            activation="swish",
            max_z: int = 100,
            weight_init: str = "xavier_uniform",
            bias_init: str = "zeros",
            **kwargs,
    ):
        """
        Initialize SaVeNet.
        """
        super(SaVeNet, self).__init__()
        seed_everything(42, workers=True)
        self.eps = 1e-8
        self.hidden_dim = hidden_dim
        self.num_encoder = num_encoder
        self.vector_alpha = 1.0

        self.omega = CosineCutoff(cutoff, 0)
        self.nonlinear_bias = nn.ParameterList(
            [
                Parameter(torch.zeros(hidden_dim))
                for _ in range(self.num_encoder)
            ]
        )  # message_bias_non, update_bias_non

        radial_basis = basis(r_basis)

        self.chi = radial_basis(cutoff=cutoff, n_rbf=num_rbf)
        self.embedding = nn.Embedding(max_z, hidden_dim, padding_idx=0)
        self.vector_embedding = nn.Linear(hidden_dim, hidden_dim * 3)

        activation, bias_init, weight_init = _init(activation, bias_init, weight_init)
        DenseInit = partial(Dense, weight_init=weight_init, bias=bias_init)
        MLPInit = partial(MLP, weight_init=weight_init, bias=bias_init)

        self.phi_chi_gamma = DenseInit(num_rbf, self.num_encoder * 3 * self.hidden_dim)

        self.phi_b = DenseInit(3, hidden_dim, bias=False)

        self.inter_atomic_comm = nn.ModuleList(
            [
                MLPInit(
                    hidden_dims=[hidden_dim, hidden_dim, 3 * hidden_dim],
                    activation=activation,
                )
                for _ in range(self.num_encoder)
            ]
        )
        self.atom_wise_comm = nn.ModuleList(
            [
                MLPInit(
                    hidden_dims=[
                        hidden_dim + hidden_dim,
                        hidden_dim,
                        hidden_dim + hidden_dim,
                    ],
                    activation=activation,
                )
                for _ in range(self.num_encoder)
            ]
        )
        self.vector_update = nn.ModuleList(
            [
                DenseInit(
                    hidden_dim,
                    hidden_dim,
                    bias=False,
                )
                for _ in range(self.num_encoder)
            ]
        )

    def forward(self, batch_data):
        """
        The forward function implements the process of the forward pass through the SaVeNet model.
        It takes as input a batch of molecule data and processes it to return the scalar and vector representations.

        The process can be described as follows:

        1. The input is passed through the initialization layers, where the geometric features'
           latent representation is learned.

        2. These latent representations are then passed to the inter-atomic interaction function
           which encapsulates inter-atomic interactions by integrating both equivariant direction vectors
           and invariant distance filters, encoded using radial basis functions.

        3. The outputs of the interaction function are then processed by atom-wise blocks,
           which compute the interaction between invariant and equivariant representations and perform channel-wise updates.

        4. The results are then passed through the decoder layers where the learned latent representations
           are decoded for downstream tasks on either invariant or equivariant targets.

        Args:
        batch_data (Data): Batch of molecule data.

        Returns:
        Data: Batch that includes scalar and vector representations.
        """
        atom, pos, batch = batch_data.z.long(), batch_data.pos, batch_data.batch
        rij, d_ij = batch_data.rij, batch_data.dir_ij

        edge_index = batch_data.edge_index
        j, i = edge_index

        t_ij = torch.cross(pos[j], pos[i])
        o_ij = torch.cross(d_ij, t_ij)
        edge_vectors = torch.stack([d_ij, t_ij, o_ij], dim=-1)
        beta_ij = self.phi_b(edge_vectors)

        phi_r = self.phi_chi_gamma(self.chi(rij) * self.omega(rij).unsqueeze(-1))
        phi_r = torch.split(phi_r, 3 * self.hidden_dim, dim=-1)

        s = self.embedding(atom)
        spherical = self.vector_embedding(s).reshape(
            s.shape[0], -1, s.shape[-1]
        )

        V = self.vector_embed_dropout(
            torch.stack(
                (
                    torch.sin(spherical[:, 0, :]) * torch.cos(spherical[:, 1, :]),
                    torch.sin(spherical[:, 0, :]) * torch.sin(spherical[:, 1, :]),
                    torch.cos(spherical[:, 0, :]),
                ), dim=1)
            * spherical[:, 2, :].unsqueeze(1)
            * self.vector_alpha
        )

        for l_e in range(self.num_encoder):
            phi_s_h_j = phi_r[l_e] * self.inter_atomic_comm[l_e](s)[j]
            phi_b_d_v, phi_d, phi_v = torch.split(phi_s_h_j, self.hidden_dim, dim=-1)
            vec_update = phi_d.unsqueeze(-2) * beta_ij + (
                    phi_v.unsqueeze(-2) * INV_SQRT_3 * V[j]
            )
            merged = flatten(phi_b_d_v, vec_update)
            merged_sum = scatter(merged, i, dim=0, reduce="sum", dim_size=s.shape[0])
            phi_b_d_v, vec_update = recover(merged_sum, vec_update.shape[-1])

            V = V + vec_update
            s = s + phi_b_d_v

            updated_v = self.vector_update[l_e](V)
            v_norm = safe_norm(updated_v, dim=-2, eps=self.eps)
            reps = torch.cat([s, v_norm], dim=-1)
            phi_vm = self.atom_wise_comm[l_e](reps)

            norm = safe_norm(updated_v, dim=1, keepdim=True, eps=self.eps)
            norm = norm + self.nonlinear_bias[l_e]
            act = F.sigmoid(norm)
            v_prime = updated_v * act
            v_prime = phi_vm[..., self.hidden_dim:].unsqueeze(1) * v_prime
            s, V = s + phi_vm[..., : self.hidden_dim], V + v_prime

        batch.s, batch.V = s, V
        return batch


class SaVeNetWrapper(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            num_encoder: int = 8,
            num_rbf: int = 20,
            cutoff: float = 5.0,
            r_basis: str = "BesselBasis",
            activation="swish",
            max_z: int = 100,
            weight_init: str = "xavier_uniform",
            bias_init: str = "zeros",
            decoder=None,
            target_mean=None,
            target_std=None,
            atomref=None
    ):
        """
        Initialize SaVeNetWrapper.

        The wrapper consists of an encoder part, the SaVeNet, and a decoder part.
        The SaVeNet is responsible for encoding the molecular structure into a latent representation, while
        the decoder translates these representations into the target properties.

        Args:
            hidden_dim (int): Dimension of the hidden layer.
            num_encoder (int): Number of encoder layers.
            num_rbf (int): Number of Radial Basis Functions for the distance encoding.
            cutoff (float): Cutoff distance for the local environment.
            r_basis (str): Radial basis type.
            activation (str): Activation function to use. Default is "swish".
            max_z (int): Maximum atomic number that the model will be applied to.
            weight_init (str): Weight initialization method.
            bias_init (str): Bias initialization method.
            decoder (nn.Module, optional): Decoder module to use. If None, a suitable default will be chosen.
            target_mean (torch.Tensor, optional): Mean of target property. Used for de-normalizing the prediction.
            target_std (torch.Tensor, optional): Standard deviation of target property. Used for de-normalizing the prediction.
            atomref (torch.Tensor, optional): Reference single-atom properties. Used for BAC.
        """
        self.cutoff = cutoff
        self.encoder = SaVeNet(hidden_dim=hidden_dim, num_encoder=num_encoder, num_rbf=num_rbf, cutoff=cutoff,
                               r_basis=r_basis, activation=activation, max_z=max_z,
                               weight_init=weight_init, bias_init=bias_init)

        if decoder is None:
            self.decoder = dec.Decoder(
                input_dims=hidden_dim,
                output_dims=1,
                num_layers=2,
                activation=activation,
                property_name="y",
                mean=target_mean,
                stddev=target_std,
                atom_references=atomref
            )

    def forward(self, batch):
        get_geometry(batch, cutoff=self.cutoff)
        encoded = self.encoder(batch)
        output = self.decoder(encoded)
        return output

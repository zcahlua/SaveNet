from typing import Optional

import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn

from .layers import Dense, MLP, NormalizedProperty, SelectProperty, str2act


class EquivariantDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, activation=F.silu, final_activation=None):
        super().__init__()
        self.out_dim = out_dim
        self.vector_comm = Dense(in_dim, out_dim, activation=None, bias=False)
        self.atom_wise_comm = nn.Sequential(
            Dense(in_dim + out_dim, hidden_dim, activation=activation),
            Dense(hidden_dim, out_dim * 2, activation=None),
        )
        self.final_activation = final_activation

    def forward(self, scalar, vector):
        comm_vector = self.vector_comm(vector)
        reps = torch.cat([scalar, torch.norm(comm_vector, dim=-2)], dim=-1)
        x = self.atom_wise_comm(reps)
        scalar_out, x = torch.split(x, [self.out_dim, self.out_dim], dim=-1)
        if self.final_activation is not None:
            scalar_out = self.final_activation(scalar_out)
        return scalar_out, x.unsqueeze(-2) * comm_vector


class Decoder(nn.Module):
    def __init__(self, input_dims, output_dims=1, num_layers=2, hidden_dims=None, activation=F.silu, property_name="y",
                 mean=None, stddev=None, atom_references=None, custom_decoder=None, position_contribution=False,
                 return_vector_key=None, aggregation_function: Optional[str] = "sum"):
        super().__init__()
        if isinstance(activation, str):
            activation = str2act(activation)
        hidden_dims = input_dims if hidden_dims is None else hidden_dims
        self.return_vector_key = return_vector_key
        self.property_name = property_name
        self.position_contribution = position_contribution
        self.aggregation_function = aggregation_function
        self.standardize = NormalizedProperty(
            torch.as_tensor([0.0] if mean is None else mean, dtype=torch.float32),
            torch.as_tensor([1.0] if stddev is None else stddev, dtype=torch.float32),
        )
        self.atom_references = nn.Embedding.from_pretrained(atom_references.type(torch.float32)) if atom_references is not None else None

        self.is_derived = custom_decoder == "derived"
        if custom_decoder is None:
            self.output_network = nn.Sequential(
                SelectProperty("s"),
                MLP([input_dims, hidden_dims, output_dims], n_layers=num_layers, activation=activation),
            )
        elif self.is_derived:
            self.output_network = nn.ModuleList([
                EquivariantDecoderBlock(in_dim=input_dims, out_dim=hidden_dims, hidden_dim=hidden_dims,
                                        activation=activation, final_activation=activation),
                EquivariantDecoderBlock(in_dim=hidden_dims, out_dim=1, hidden_dim=hidden_dims, activation=activation),
            ])
        else:
            self.output_network = custom_decoder

    def forward(self, inputs):
        result = {}
        atoms = inputs.z

        if self.is_derived:
            s, V = inputs.s, inputs.V
            for layer in self.output_network:
                s, V = layer(s, V)
            yi = torch.squeeze(V, -1) + (inputs.pos * s) if self.position_contribution else s
            if self.return_vector_key:
                result[self.return_vector_key] = V
        else:
            yi = self.output_network(inputs)

        if self.atom_references is not None:
            yi = yi + self.atom_references(atoms)
        if self.aggregation_function is not None:
            yi = torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_function)
        yi = self.standardize(yi)

        result[self.property_name] = yi
        return result

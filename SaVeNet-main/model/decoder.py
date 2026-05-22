from typing import Optional

import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn

from .layers import Dense, NormalizedProperty, SelectProperty, MLP


class EquivariantDecoderBlock(nn.Module):

    def __init__(
            self, in_dim, out_dim, hidden_dim, activation=F.silu, final_activation=None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Vector communication layer
        self.vector_comm = Dense(in_dim, out_dim, activation=None, bias=False)

        # Atom-wise communication layer
        self.atom_wise_comm = nn.Sequential(
            Dense(in_dim + out_dim, hidden_dim, activation=activation),
            Dense(hidden_dim, out_dim * 2, activation=None),
        )

        self.final_activation = final_activation

    def forward(self, scalar, vector):
        # Pass vector through the vector communication layer
        comm_vector = self.vector_comm(vector)

        # Concatenate scalar and norm of the communication vector
        reps = torch.cat([scalar, torch.norm(comm_vector, dim=-2)], dim=-1)

        # Pass the context through the atom-wise communication layer
        x = self.atom_wise_comm(reps)

        # Split the output
        scalar_out, x = torch.split(x, [self.out_dim, self.out_dim], dim=-1)

        # Apply final activation if present
        final_scalar_out = (
            self.final_activation(scalar_out) if self.final_activation else scalar_out
        )

        return final_scalar_out, x.unsqueeze(-2) * comm_vector


import torch
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims=1,
        num_layers=2,
        hidden_dims=None,
        activation=F.silu,
        property_name="y",
        mean=None,
        stddev=None,
        atom_references=None,
        custom_decoder=None,
        position_contribution=False,
        return_vector_key=None,
        aggregation_function: Optional[str] = "sum",
    ):
        super(AtomwiseDecoder, self).__init__()
        self.setup_variables(
            input_dims, output_dims, num_layers, hidden_dims, activation,
            property_name, mean, stddev, atom_references, custom_decoder,
            position_contribution, return_vector_key, aggregation_function
        )
        self.initialize_output_network(
            custom_decoder, input_dims, hidden_dims, output_dims, num_layers, activation
        )

    def setup_variables(
        self, input_dims, output_dims, num_layers, hidden_dims, activation,
        property_name, mean, stddev, atom_references, custom_decoder,
        position_contribution, return_vector_key, aggregation_function
    ):
        self.return_vector_key = return_vector_key
        self.num_layers = num_layers
        self.property_name = property_name
        self.position_contribution = position_contribution
        self.aggregation_function = aggregation_function

        self.standardize = self.create_standardization_layer(
            mean, stddev
        )

        self.atom_references = self.create_atom_reference_layer(
            atom_references
        )

        self.is_derived = False

    def create_standardization_layer(self, mean, stddev):
        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev
        return NormalizedProperty(mean, stddev) if mean is not None and stddev is not None else nn.Identity()

    def create_atom_reference_layer(self, atom_references):
        if atom_references is not None:
            return nn.Embedding.from_pretrained(atom_references.type(torch.float32))
        return None

    def initialize_output_network(
        self, custom_decoder, input_dims, hidden_dims, output_dims, num_layers, activation
    ):
        if custom_decoder is None:
            self.output_network = nn.Sequential(
                SelectProperty("s"),
                MLP([input_dims, hidden_dims, output_dims], num_layers=num_layers, activation=activation),
            )
        elif custom_decoder == "derived":
            self.setup_derived_output_network(input_dims, hidden_dims, activation)
        else:
            self.output_network = custom_decoder

    def setup_derived_output_network(self, input_dims, hidden_dims, activation):
        hidden_dims = input_dims if hidden_dims is None else hidden_dims
        self.output_network = nn.ModuleList([
            EquivariantDecoderBlock(
                in_dim=input_dims, out_dim=hidden_dims, hidden_dim=hidden_dims, activation=activation, final_activation=activation
            ),
            EquivariantDecoderBlock(
                in_dim=hidden_dims, out_dim=1, hidden_dim=hidden_dims, activation=activation
            ),
        ])
        self.is_derived = True

    def forward_derived(self, inputs, result):
        s, V = inputs.s, inputs.V
        for layer in self.output_network:
            s, V = layer(s, V)

        if self.position_contribution:
            atomic_dipoles = torch.squeeze(V, -1)
            charges = s
            dipole_offsets = inputs.pos * charges
            yi = atomic_dipoles + dipole_offsets
        else:
            yi = s

        if self.return_vector_key:
            result[self.return_vector_key] = V

        result[self.property_name] = self.standardize(yi)
        return result

    def forward_default(self, inputs, result):
        yi = self.output_network(inputs)
        yi = self.standardize(yi)
        result[self.property_name] = yi
        return result

    def forward(self, inputs):
        atoms = inputs.z
        result = {}

        if self.is_derived:
            result = self.forward_derived(inputs, result)
        else:
            result = self.forward_default(inputs, result)

        if self.atom_references is not None:
            result[self.property_name] += self.atom_references(atoms)

        if self.aggregation_function is not None:
            result[self.property_name] = torch_scatter.scatter(
                result[self.property_name], inputs.batch, dim=0, reduce=self.aggregation_function
            )

        return result

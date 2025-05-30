# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Sequence, Union

import torch
from torch import nn

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)


class FullyConnectedTensorProductConv(nn.Module):
    r"""
    Message passing layer for tensor products in DiffDock-like architectures.
    The left operand of tensor product is the node features; the right operand
    consists of the spherical harmonic of edge vector.

    Mathematical formulation:

    .. math::

        \sum_{b \in \mathcal{N}_a} \mathbf{h}_b \otimes_{\psi_{a b}} Y\left(\hat{r}_{a b}\right)

    where the path weights :math:`\psi_{a b}` can be constructed from edge
    embeddings and scalar features using an MLP:

    .. math::

        \psi_{a b} = \operatorname{MLP} \left(e_{a b}, \mathbf{h}_a^0, \mathbf{h}_b^0\right)

    Users have the option to either directly input the weights or provide the
    MLP parameters and scalar features from edges and nodes.

    Args:
        in_irreps (Irreps): Irreps for the input node features.
        sh_irreps (Irreps): Irreps for the spherical harmonic representations of edge vectors.
        out_irreps (Irreps): Irreps for the output.
        batch_norm (bool, optional): If true, batch normalization is applied. Defaults to True.
        mlp_channels (Sequence of int, optional): A sequence of integers defining the number of neurons in each layer in MLP before the output layer. If None, no MLP will be added. The input layer contains edge embeddings and node scalar features. Defaults to None.
        mlp_activation (``nn.Module`` or Sequence of ``nn.Module``, optional): A sequence of functions to be applied in between linear layers in MLP, e.g., ``nn.Sequential(nn.ReLU(), nn.Dropout(0.4))``. Defaults to ``nn.GELU()``.
        layout (IrrepsLayout, optional): The layout of the input and output irreps. Default is ``cue.mul_ir`` which is the layout corresponding to e3nn.
        use_fallback (bool, optional): If `None` (default), a CUDA kernel will be used if available.
                If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
                If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

    Examples:
        >>> in_irreps = cue.Irreps("O3", "4x0e + 4x1o")
        >>> sh_irreps = cue.Irreps("O3", "0e + 1o")
        >>> out_irreps = cue.Irreps("O3", "4x0e + 4x1o")

        **Case 1**: MLP with the input layer having 6 channels and 2 hidden layers
        having 16 channels. edge_emb.size(1) must match the size of the input layer: 6

        >>> conv1 = FullyConnectedTensorProductConv(in_irreps, sh_irreps, out_irreps,
        ...     mlp_channels=[6, 16, 16], mlp_activation=nn.ReLU(), layout=cue.ir_mul)
        >>> conv1
        FullyConnectedTensorProductConv(...)
        >>> # out = conv1(src_features, edge_sh, edge_emb, graph)

        **Case 2**: If edge_emb is constructed by concatenating scalar features from
        edges, sources and destinations, as in DiffDock, the layer can accept each
        scalar component separately:

        >>> # out = conv1(src_features, edge_sh, edge_emb, graph, src_scalars, dst_scalars)

        This allows a smaller GEMM in the first MLP layer by performing GEMM on each
        component before indexing. The first-layer weights are split into sections
        for edges, sources and destinations, in that order. This is equivalent to

        >>> # src, dst = graph.edge_index
        >>> # edge_emb = torch.hstack((edge_scalars, src_scalars[src], dst_scalars[dst]))
        >>> # out = conv1(src_features, edge_sh, edge_emb, graph)

        **Case 3**: No MLP, edge_emb will be directly used as the tensor product weights:

        >>> conv3 = FullyConnectedTensorProductConv(in_irreps, sh_irreps, out_irreps,
        ...     mlp_channels=None, layout=cue.ir_mul)
        >>> # out = conv3(src_features, edge_sh, edge_emb, graph)
    """

    def __init__(
        self,
        in_irreps: cue.Irreps,
        sh_irreps: cue.Irreps,
        out_irreps: cue.Irreps,
        batch_norm: bool = True,
        mlp_channels: Optional[Sequence[int]] = None,
        mlp_activation: Union[nn.Module, Sequence[nn.Module], None] = nn.GELU(),
        layout: cue.IrrepsLayout = None,  # e3nn_compat_mode
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()

        in_irreps, out_irreps, sh_irreps = default_irreps(
            in_irreps, out_irreps, sh_irreps
        )
        assert_same_group(in_irreps, out_irreps, sh_irreps)

        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.layout = default_layout(layout)

        self.tp = cuet.FullyConnectedTensorProduct(
            in_irreps,
            sh_irreps,
            out_irreps,
            layout=self.layout,
            shared_weights=False,
            use_fallback=use_fallback,
        )

        self.batch_norm = (
            cuet.layers.BatchNorm(out_irreps, layout=self.layout)
            if batch_norm
            else None
        )

        if mlp_activation is None:
            mlp_activation = []
        elif hasattr(mlp_activation, "__len__") and hasattr(
            mlp_activation, "__getitem__"
        ):
            mlp_activation = list(mlp_activation)
        else:
            mlp_activation = [mlp_activation]

        if mlp_channels is not None:
            dims = list(mlp_channels) + [self.tp.weight_numel]
            mlp = []
            for i in range(len(dims) - 1):
                mlp.append(nn.Linear(dims[i], dims[i + 1]))
                if i != len(dims) - 2:
                    mlp.extend(mlp_activation)
            self.mlp = nn.Sequential(*mlp)
        else:
            self.mlp = None

    def forward(
        self,
        src_features: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_emb: torch.Tensor,
        graph: tuple[torch.Tensor, tuple[int, int]],
        src_scalars: Optional[torch.Tensor] = None,
        dst_scalars: Optional[torch.Tensor] = None,
        reduce: str = "mean",
        edge_envelope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src_features (torch.Tensor): Source node features.
                Shape: (num_src_nodes, in_irreps.dim)
            edge_sh (torch.Tensor): The spherical harmonic representations of the edge vectors.
                Shape: (num_edges, sh_irreps.dim)
            edge_emb (torch.Tensor): Edge embeddings that are fed into MLPs to generate tensor product weights.
                Shape: (num_edges, dim), where `dim` should be:

                - `tp.weight_numel` when the layer does not contain MLPs.
                - num_edge_scalars, when scalar features from edges, sources and destinations are passed in separately.
            graph (tuple): A tuple that stores the graph information, with the first element being the adjacency matrix in COO, and the second element being its shape:
                (num_src_nodes, num_dst_nodes)
            src_scalars (torch.Tensor, optional): Scalar features of source nodes. See examples for usage.
                Shape: (num_src_nodes, num_src_scalars)
            dst_scalars (torch.Tensor, optional): Scalar features of destination nodes. See examples for usage.
                Shape: (num_dst_nodes, num_dst_scalars)
            reduce (str, optional): Reduction operator. Choose between "mean" and "sum". Defaults to "mean".
            edge_envelope (torch.Tensor, optional): Typically used as attenuation factors to fade out messages coming from nodes close to the cutoff distance used to create the graph. This is important to make the model smooth to the changes in node's coordinates.
                Shape: (num_edges,)

        Returns:
            torch.Tensor: Output node features. Shape: (num_dst_nodes, out_irreps.dim)
        """
        edge_emb_size = edge_emb.size(-1)
        src_scalars_size = 0 if src_scalars is None else src_scalars.size(-1)
        dst_scalars_size = 0 if dst_scalars is None else dst_scalars.size(-1)

        if self.mlp is None:
            if self.tp.weight_numel != edge_emb_size:
                raise RuntimeError(
                    f"When MLP is not present, edge_emb's last dimension must "
                    f"equal tp.weight_numel (but got {edge_emb_size} and "
                    f"{self.tp.weight_numel})"
                )
        else:
            total_size = edge_emb_size + src_scalars_size + dst_scalars_size
            if self.mlp[0].in_features != total_size:
                raise RuntimeError(
                    f"The size of MLP's input layer ({self.mlp[0].in_features}) "
                    f"does not match the total number of scalar features from "
                    f"edge_emb, src_scalars and dst_scalars ({total_size})"
                )

        if reduce not in ["mean", "sum"]:
            raise RuntimeError(
                f"reduce argument must be either 'mean' or 'sum', got {reduce}."
            )

        (src, dst), (num_src_nodes, num_dst_nodes) = graph

        if self.mlp is not None:
            if src_scalars is None and dst_scalars is None:
                tp_weights = self.mlp(edge_emb)
            else:
                w_edge, w_src, w_dst = torch.split(
                    self.mlp[0].weight,
                    (edge_emb_size, src_scalars_size, dst_scalars_size),
                    dim=-1,
                )
                tp_weights = edge_emb @ w_edge.T + self.mlp[0].bias

                if src_scalars is not None:
                    tp_weights += (src_scalars @ w_src.T)[src]

                if dst_scalars is not None:
                    tp_weights += (dst_scalars @ w_dst.T)[dst]

                tp_weights = self.mlp[1:](tp_weights)
        else:
            tp_weights = edge_emb

        out = self.tp(src_features[src], edge_sh, tp_weights)

        if edge_envelope is not None:
            out = out * edge_envelope.view(-1, 1)

        out = scatter_reduce(out, dst, dim=0, dim_size=num_dst_nodes, reduce=reduce)

        if self.batch_norm:
            out = self.batch_norm(out)

        return out


def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
    size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


def scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,  # value of out.size(dim)
    reduce: str = "sum",  # "sum", "prod", "mean", "amax", "amin"
):
    # scatter() expects index to be int64
    index = broadcast(index, src, dim).to(torch.int64)

    size = list(src.size())

    if dim_size is not None:
        size[dim] = dim_size
    else:
        size[dim] = 0 if index.numel() == 0 else int(index.max()) + 1

    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(dim, index, src, reduce, include_self=False)

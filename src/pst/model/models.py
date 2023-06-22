from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GraphNorm

from .layers import MultiheadAttentionConv, MultiheadAttentionPooling
from ._types import OptionalAttentionOutput


class SetTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        n_enc_layers: int = 2,
        multiplier: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # the pyg strategy for attention is for each head to attend to the full
        # length of the input, rather than splitting the input into `head`-sized
        # chunks. The memory reqs can be bypassed by projecting to a smaller
        # dim that is desired out_dim // heads, and then concatenated the heads
        # back together.
        hidden_dim, remainder = divmod(out_dim, heads)
        if remainder != 0:
            raise ValueError(f"{out_dim=} must be divisible by {heads=}")

        ##### ENCODER #####
        self._encoder = nn.ModuleDict()
        encoder_layers = nn.ModuleList()
        start_dim = in_dim
        for _ in range(n_enc_layers):
            layer = MultiheadAttentionConv(
                in_channels=start_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
            encoder_layers.append(layer)
            start_dim = out_dim

        self._encoder["layers"] = encoder_layers

        # TODO: only add if doing pre-normalization
        self._encoder["norm"] = GraphNorm(
            in_channels=out_dim,
        )
        # before encoding: [N, D]
        # where N is the total number of items in each set
        # or the total nodes in all graphs
        # D is the original embedding dim
        # after encoder: [N, D']
        # D' is the hidden dim
        ###################

        ##### DECODER #####
        self._decoder = nn.ModuleDict()
        # shape: [N, D'] -> [B, D'] where B is the batch size, ie
        # number of graphs/sets
        self._decoder["pool"] = MultiheadAttentionPooling(
            in_channels=out_dim,
            heads=heads,
            multiplier=multiplier,
            dropout=dropout,
        )
        self._decoder["linear"] = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Linear(out_dim, out_dim),
        )
        # final shapes: [B, D'] -> [B, D''], ie output dimension now
        ###################

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, D] -> [N, D']
        for layer in self._encoder["layers"]:  # type: ignore
            x = layer(x=x, edge_index=edge_index, return_attention_weights=False)

        x = self._encoder["norm"](x)

        return x

    def decode(
        self,
        x_out: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptionalAttentionOutput:
        x_avg = self._decoder["pool"](
            x=x_out,
            edge_index=edge_index,
            ptr=ptr,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            x_avg, (forward_edge_index, attn) = x_avg

        output = self._decoder["linear"](x_avg)

        if return_attention_weights:
            return output, (forward_edge_index, attn)  # type: ignore
        return output

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptionalAttentionOutput:
        x_out = self.encode(x=x, edge_index=edge_index)
        return self.decode(
            x_out=x_out,
            edge_index=edge_index,
            ptr=ptr,
            return_attention_weights=return_attention_weights,
        )

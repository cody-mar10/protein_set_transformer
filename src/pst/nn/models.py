from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GraphNorm

from pst.nn.layer_drop import LayerDropModuleList
from pst.nn.layers import (
    MultiheadAttentionConv,
    MultiheadAttentionPooling,
    PositionwiseFeedForward,
)
from pst.typing import OptGraphAttnOutput


class SetTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        n_layers: int,
        dropout: float,
        layer_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.layers = LayerDropModuleList(layer_dropout)
        start_dim = in_dim
        for _ in range(n_layers):
            layer = MultiheadAttentionConv(
                in_channels=start_dim,
                out_channels=out_dim,
                heads=num_heads,
                dropout=dropout,
            )
            self.layers.append(layer)
            start_dim = out_dim

        self.final_norm = GraphNorm(in_channels=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            # TODO: what if returning attn weights
            x = layer(
                x=x,
                edge_index=edge_index,
                batch=batch,
                return_attention_weights=return_attention_weights,
            )

        x = self.final_norm(x, batch)
        return x


class SetTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.pool = MultiheadAttentionPooling(
            feature_dim=hidden_dim, heads=num_heads, dropout=dropout
        )

        self.linear = PositionwiseFeedForward(
            in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        ptr: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptGraphAttnOutput:
        pooled_x, attn = self.pool(
            x=x,
            ptr=ptr,
            batch=batch,
            return_attention_weights=return_attention_weights,
        )
        output = self.linear(pooled_x)
        return output, attn


class SetTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        n_enc_layers: int,
        dropout: float,
        layer_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = SetTransformerEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )

        self.decoder = SetTransformerDecoder(
            hidden_dim=out_dim, num_heads=num_heads, dropout=dropout
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        # x: [N, D] -> [N, D']
        return self.encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            return_attention_weights=return_attention_weights,
        )

    def decode(
        self,
        x_out: torch.Tensor,
        ptr: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptGraphAttnOutput:
        # shape: [N, D'] -> [B, D']
        return self.decoder(
            x=x_out,
            ptr=ptr,
            batch=batch,
            return_attention_weights=return_attention_weights,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptGraphAttnOutput:
        x_out = self.encode(
            x=x,
            edge_index=edge_index,
            batch=batch,
            return_attention_weights=False,
        )

        graph_rep, attn = self.decode(
            x_out=x_out,
            ptr=ptr,
            batch=batch,
            return_attention_weights=return_attention_weights,
        )
        return graph_rep, attn

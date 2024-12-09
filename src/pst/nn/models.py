import torch
from torch import nn
from torch_geometric.nn import MLP, GraphNorm

from pst.nn.layer_drop import LayerDropModuleList
from pst.nn.layers import MultiheadAttentionConv, MultiheadAttentionPooling
from pst.typing import EdgeAttnOutput, GraphAttnOutput, OptTensor


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
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

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
                add_self_loops=False,
            )
            self.layers.append(layer)
            start_dim = out_dim

        self.final_norm = GraphNorm(in_channels=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> EdgeAttnOutput:
        for layer in self.layers:
            out: EdgeAttnOutput = layer(
                x=x,
                edge_index=edge_index,
                batch=batch,
                node_mask=node_mask,
                return_attention_weights=return_attention_weights,
            )

            x = out.out

        x = self.final_norm(x, batch)

        if return_attention_weights:
            final_output = EdgeAttnOutput(
                out=x, edge_index=out.edge_index, attn=out.attn
            )
        else:
            final_output = EdgeAttnOutput(out=x, edge_index=None, attn=None)

        return final_output


class SetTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.actv = nn.GELU()
        self.pool = MultiheadAttentionPooling(
            feature_dim=hidden_dim, heads=num_heads, dropout=dropout
        )
        self.mlp = MLP(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=2,
            dropout=dropout,
            act=self.actv,
            norm=None,
            plain_last=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        ptr: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> GraphAttnOutput:
        # the impl from the SetTransformer paper says this occurs first before MH attn
        # but this is missing from the paper's code impl
        x = self.actv(self.linear(x))

        pooled_x, attn = self.pool(
            x=x,
            ptr=ptr,
            batch=batch,
            return_attention_weights=return_attention_weights,
        )
        output = self.mlp(pooled_x)
        return GraphAttnOutput(out=output, attn=attn)


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
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> EdgeAttnOutput:
        # x: [N, D] -> [N, D']
        return self.encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            node_mask=node_mask,
            return_attention_weights=return_attention_weights,
        )

    def decode(
        self,
        x_out: torch.Tensor,
        ptr: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> GraphAttnOutput:
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
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> GraphAttnOutput:
        x_out, *_ = self.encode(
            x=x,
            edge_index=edge_index,
            batch=batch,
            node_mask=node_mask,
            return_attention_weights=False,
        )

        graph_out = self.decode(
            x_out=x_out,
            ptr=ptr,
            batch=batch,
            return_attention_weights=return_attention_weights,
        )
        return graph_out

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn
from torch_geometric.nn import GraphNorm, MessagePassing
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, segment, softmax

from pst.typing import OptEdgeAttnOutput, OptGraphAttnOutput


class MultiheadAttentionConv(MessagePassing):
    """Implement the SetTransformer attention layer using a graph-level
    implementation. Graph implementations are useful when the number of items
    greatly varies among sets, resulting in substantial memory and computation
    savings.
    """

    _alpha: OptTensor

    # TODO: add option to change normalization strategy <- actually don't think it
    # matters that much
    # the major bottleneck was with chamfer dist calc
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        concat: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Create a multihead attention layer that operates on graph-structured
        data. This performs identical calculations to pre-normalization
        transformers.

        Args:
            in_channels (int): Input embedding dimension.
            out_channels (int): Output embedding dimension. Recommended value
                is for this to be `:param:in_channels` // `:param:heads` with
                `:param:concat=True` so that the effective output dimension
                remains the same as the input.
            heads (int): Number of attention heads.
            concat (bool, optional): Concatenate attention heads if True,
                Average over attention heads if False. Defaults to True.
            dropout (float, optional): Dropout fraction for the final
                linear layer. Defaults to 0.0.
            **kwargs: Additional arguments for the `MessagePassing` base class
        """
        # this setup mirrors exactly how attention would be computed normally
        super().__init__(aggr="add", node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.eff_out_channels = heads * out_channels
        self.concat = concat
        self.scale = math.sqrt(self.out_channels)
        self.dropout = dropout
        self._alpha = None

        self.linQ = nn.Linear(in_channels, self.eff_out_channels)
        self.linK = nn.Linear(in_channels, self.eff_out_channels)
        self.linV = nn.Linear(in_channels, self.eff_out_channels)
        self.normQ = GraphNorm(in_channels)

        if concat:
            # aggr_out will have shape: [-1, H * D]
            self.normO = GraphNorm(self.eff_out_channels)
            # self.linO = nn.Linear(self.eff_out_channels, self.eff_out_channels)
            self.linO = PositionwiseFeedForward(
                hidden_dim=self.eff_out_channels,
                out_dim=self.eff_out_channels,
                dropout=self.dropout,
            )
        else:
            # aggr_out will have shape: [-1, D]
            self.normO = GraphNorm(self.out_channels)
            # self.linO = nn.Linear(out_channels, out_channels)
            self.linO = PositionwiseFeedForward(
                hidden_dim=self.out_channels,
                out_dim=self.out_channels,
                dropout=self.dropout,
            )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.linQ.reset_parameters()
        self.linK.reset_parameters()
        self.linV.reset_parameters()
        self.linO.reset_parameters()
        self.normQ.reset_parameters()
        self.normO.reset_parameters()

    def uncat_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "nodes (heads dim) -> nodes heads dim",
            heads=self.heads,
            dim=self.out_channels,
        )

    def cat_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "nodes heads dim -> nodes (heads dim)")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptEdgeAttnOutput:
        """Forward pass to compute scaled-dot product attention to update each
        node/item representation.

        Args:
            x (torch.Tensor): Stacked node/item feature tensor [N, D]
            edge_index (torch.Tensor): [2, E] LongTensor encoding node/item
                connectivity. To exactly mirror transformers, this is fully
                connected for each graph.
            return_attention_weights (bool, optional): Whether to return the
                attention weights and edge index. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                If `:param:return_attention_weights=False`, only return the updated
                node/item features. Otherwise, additionally return a second tuple
                that contains the adjusted `edge_index` that contains self-loops
                [2, E'] and the attention weights [E', H].
        """
        x_input = x

        # pre-normalization
        x = self.normQ(x)

        # shape [N, H * D] -> [N, H, D]
        query: torch.Tensor = self.uncat_heads(self.linQ(x))
        key: torch.Tensor = self.uncat_heads(self.linK(x))
        value: torch.Tensor = self.uncat_heads(self.linV(x))
        edge_index, _ = add_self_loops(edge_index)

        # propagate order:
        # 1. self.message
        # 2. self.aggregrate
        # 3. self.update
        out = self.propagate(
            edge_index=edge_index, initial=x_input, query=query, key=key, value=value
        )

        # get attn weights
        alpha = self._pop_alpha()

        if return_attention_weights:
            return out, (edge_index, alpha)

        return out

    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        index: torch.Tensor,
        ptr: OptTensor = None,
        size_i: Optional[int] = None,
    ) -> torch.Tensor:
        # input shapes: [E, H, D]

        # this computes scaled dot-product attention
        # inner term during self-attention

        # shape: [E, H]
        qk = torch.sum(query_i * key_j, -1) / self.scale

        # shape: [E, H]
        alpha = softmax(qk, index=index, ptr=ptr, num_nodes=size_i, dim=0)

        # store attn weights
        self._push_alpha(alpha)

        # shape: [E, H, D]
        out = value_j * rearrange(alpha, "edges heads -> edges heads 1")
        return out

    def update(self, aggr_out: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        if self.concat:
            # shape: [-1, H, D] -> [-1, H * D]
            aggr_out = self.cat_heads(aggr_out)
        else:
            # shape: [-1, H, D] -> [-1, D]
            aggr_out = reduce(aggr_out, "nodes heads dim -> nodes dim", "mean")

        # initial shape: [-1, Di] != [-1, H * D] necessarily
        # this doesn't matter if you just set the out dim D to be
        # Di // H like usual, but matters for projecting to 1 dim
        # during pooling -- similar problem below
        out_dim = aggr_out.size(-1)
        init_dim = initial.size(-1)
        if init_dim != out_dim:
            # basically adding all features from initial to aggr_out
            leftover = init_dim // out_dim
            initial = reduce(
                initial,
                "nodes (left dim) -> nodes dim",
                reduction="sum",
                left=leftover,
                dim=out_dim,
            )

        # residuals before normalization, ie pre-norm
        aggr_out = aggr_out + initial
        normed_aggr_out = self.normO(aggr_out)

        # linO is a 2-layer feedforward network
        aggr_out = aggr_out + self.linO(normed_aggr_out)

        return aggr_out

    # these two methods are convenience methods to pass around the attn weights
    # since the PyG MessagePassing framework doesn't explicitly allow
    # for returning multiple values from the self.propagate method
    def _push_alpha(self, alpha: torch.Tensor):
        self._alpha = alpha.detach()

    def _pop_alpha(self) -> torch.Tensor:
        # assert self._alpha is None
        alpha: torch.Tensor = self._alpha  # type: ignore
        self._alpha = None
        return alpha


# Modeled based off the SAGPooling operation
class MultiheadAttentionPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads: int,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a multihead attention pooling module. This projects the
        input features onto a single dimension, and uses those scores to
        perform a weighted average.

        Args:
            in_channels (int): Input embedding dimension.
            heads (int): Number of attention heads.
            layers (int, optional): Number of GNN attention layers to use.
                Defaults to 1.
            dropout (float, optional): Dropout rate for the attention calculation.
                Defaults to 0.0.
        """
        super().__init__()
        if layers < 1:
            raise ValueError("Must have at least one layer.")

        self.layers = nn.ModuleList()
        kwargs: dict[str, Any] = dict(heads=heads, dropout=dropout, concat=False)

        in_channels = in_channels
        out_channels = in_channels // 2
        for _ in range(layers):
            self.layers.append(
                MultiheadAttentionConv(
                    in_channels=in_channels, out_channels=out_channels, **kwargs
                )
            )
            # slowy reduce size
            in_channels = out_channels
            out_channels //= 2

        # normalize final layer
        self.norm = GraphNorm(in_channels)

        # feed forward now projects to 1 dim
        self.lin = PositionwiseFeedForward(
            hidden_dim=in_channels, out_dim=1, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        batch: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> OptGraphAttnOutput:
        """Compute an attention-based score by projecting the input `x`
        onto a single dimension. These scores are then converted to probabilites
        using the softmax function, and then used to perform a weighted
        average along the inputs.

        Args:
            x (torch.Tensor): [N, D] Input stacked node/item features.
            edge_index (torch.Tensor): [2, E] LongTensor encoding node/item
                connectivity. To exactly mirror transformers, this is fully
                connected for each graph.
            ptr (torch.Tensor): Points to the start/stop of each graph/set
                in the input `:param:x` stacked features. `ptr[i]` points to
                the starting index of graph/set i, and `ptr[i+1]` points to
                the end.
            batch (torch.Tensor, optional): [N] LongTensor encoding the graph/set
                each node/item belongs to. Only required if standardizing inputs.
            return_attention_weights (bool, optional): Whether to return the
                attention weights for each node/item when computing the graph/set
                representation as a weighted average. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                If `:param:return_attention_weights=False`, just returns the
                weighted average over all nodes in graph (items in set).
                Shape [batch_size, D]. Otherwise, also returns the item-wise
                attention weights that computed the weighted average.
        """
        attn_score = x
        for layer in self.layers:
            attn_score = layer(attn_score, edge_index, return_attention_weights=False)

        # standardize score -> this helps substantially by not forcing model
        # to choose most influential node/item, instead allowing attn to more
        # than one node/item
        attn_score = self.norm(attn_score, batch)
        attn_score = self.lin(attn_score)

        # shape: [N, 1]
        weights = softmax(attn_score, ptr=ptr, dim=0)

        # x shape: [N, embedding_dim]
        # output shape: [batch_size, embedding_dim]
        weighted_avg = segment(x * weights, ptr, reduce="mean")
        if return_attention_weights:
            return weighted_avg, weights

        return weighted_avg


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        **actv_kwargs,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation(**actv_kwargs)
        self.dropout = dropout
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # following fairseq impl where no actv after 2nd linear layer
        h: torch.Tensor = self.w1(x)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.w2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_size=2048):
        super().__init__()

        self.embedding_dim = dim
        self.max_size = max_size
        self._embedding = nn.Embedding(max_size, dim)

    def expand(self, max_size: int):
        self._embedding = nn.Embedding(max_size, self.embedding_dim)

    def forward(self, positional_idx: torch.Tensor) -> torch.Tensor:
        return self._embedding(positional_idx)


class FixedPositionalEncoding(nn.Module):
    """Sinusoidal fixed positional encodings.

    Encodings are computed on the fly for each batch.
    """

    def positional_encoding(self, x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        dim = x.size(-1)
        max_size = sizes.amax().item()

        # shape: [dim / 2]
        exp = torch.arange(0, dim, 2, dtype=torch.float) / dim
        inv_freq = 1.0 / (10000**exp)

        # shape: [max_size]
        position = torch.arange(0, max_size, dtype=torch.float)

        # shape: [max_size, dim / 2]
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)

        # shape: [max_size, dim]
        enc = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1).to(x.device)
        return enc

    def forward(
        self, x: torch.Tensor, sizes: torch.Tensor, positional_idx: torch.Tensor
    ) -> torch.Tensor:
        encoding = self.positional_encoding(x, sizes)
        return encoding[positional_idx]


class ResidualMultiheadAttentionConv(MultiheadAttentionConv):
    # TODO: this module is not finished
    def forward(
        self,
        pair: PairTensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        H, D = self.heads, self.out_channels
        # x takes the post-norm route, while xres accumulates pre-norm residuals
        # shapes: [N, D]
        x, xres = pair

        query: torch.Tensor = self.linQ(x).view(-1, H, D)
        key: torch.Tensor = self.linK(x).view(-1, H, D)
        value: torch.Tensor = self.linV(x).view(-1, H, D)
        edge_index, _ = add_self_loops(edge_index)

        # propagate order:
        # 1. self.message
        # 2. self.aggregrate
        # 3. self.update
        out = self.propagate(edge_index=edge_index, query=query, key=key, value=value)

        # get attn weights
        alpha = self._pop_alpha()

        if return_attention_weights:
            return out, (edge_index, alpha)

        return out

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        if self.concat:
            # shape: [-1, H, D] -> [-1, H * D]
            aggr_out = aggr_out.view(-1, self.eff_out_channels)
        else:
            # shape: [-1, H, D] -> [-1, D]
            aggr_out = aggr_out.mean(dim=1)

        return aggr_out

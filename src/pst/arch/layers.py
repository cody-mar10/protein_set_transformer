from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphNorm, MessagePassing
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, segment, softmax

from pst._typing import OptionalAttentionOutput


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
            self.linO = nn.Linear(self.eff_out_channels, self.eff_out_channels)
        else:
            # aggr_out will have shape: [-1, D]
            self.normO = GraphNorm(self.out_channels)
            self.linO = nn.Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.linQ.reset_parameters()
        self.linK.reset_parameters()
        self.linV.reset_parameters()
        self.linO.reset_parameters()
        self.normQ.reset_parameters()
        self.normO.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptionalAttentionOutput:
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
        H, D = self.heads, self.out_channels
        x_input = x

        # pre-normalization
        x = self.normQ(x)

        query: torch.Tensor = self.linQ(x).view(-1, H, D)
        key: torch.Tensor = self.linK(x).view(-1, H, D)
        value: torch.Tensor = self.linV(x).view(-1, H, D)
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
        # this computes scaled dot-product attention
        # inner term during self-attention
        qk = torch.sum(query_i * key_j, -1) / self.scale
        alpha = softmax(qk, index=index, ptr=ptr, num_nodes=size_i, dim=0)
        # store attn weights
        self._push_alpha(alpha)
        out = value_j * alpha.view(-1, self.heads, 1)
        return out

    def update(self, aggr_out: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        if self.concat:
            # shape: [-1, H, D] -> [-1, H * D]
            aggr_out = aggr_out.view(-1, self.eff_out_channels)
        else:
            # shape: [-1, H, D] -> [-1, D]
            aggr_out = aggr_out.mean(dim=1)

        # initial shape: [-1, Di] != [-1, H * D] necessarily
        # this doesn't matter if you just set the out dim D to be
        # Di // H like usual, but matters for projecting to 1 dim
        # during pooling -- similar problem below
        out_dim = aggr_out.size(-1)
        if initial.size(-1) != aggr_out.size(-1):
            # basically adding all features from initial to aggr_out
            leftover = initial.size(-1) // out_dim
            initial = initial.view(-1, leftover, out_dim).sum(dim=1)

        # residuals before normalization, ie pre-norm
        aggr_out = aggr_out + initial
        normed_aggr_out = self.normO(aggr_out)
        aggr_out = aggr_out + F.dropout(
            F.relu(self.linO(normed_aggr_out)),
            p=self.dropout,
            training=self.training,
        )

        return aggr_out

    # these two methods are convenience methods to pass around the attn weights
    # since the PyG MessagePassing framework doesn't explicitly allow
    # for returning multiple values from the self.propagate method
    def _push_alpha(self, alpha: torch.Tensor):
        self._alpha = alpha

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
        multiplier: float = 1.0,  # TODO: this could be trainable?
        dropout: float = 0.0,
    ) -> None:
        """Initialize a multihead attention pooling module. This projects the
        input features onto a single dimension, and uses those scores to
        perform a weighted average.

        Args:
            in_channels (int): Input embedding dimension.
            heads (int): Number of attention heads.
            multiplier (float): Score multiplier. Defaults to 1.0. Values > 1.0
                will emphasize node/item importance more, while values between
                (0, 1) will de-emphasize importance and lead to more uniform
                scores like a simple average.
            dropout (float, optional): Dropout rate for the attention calculation.
                Defaults to 0.0.
        """
        super().__init__()
        self.pooling_attn_layer = MultiheadAttentionConv(
            in_channels, 1, heads=heads, concat=False, dropout=dropout
        )

        if multiplier <= 0.0:
            raise ValueError(
                f"Passed {multiplier=}. This value should be greater than 0.0"
            )

        self.multiplier = multiplier

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute an attention-based score by projecting the input `x`
        onto a single dimension. These scores are then converted probabilites
        using the softmax function, and then used to perform a weighted
        average along the inputs.

        Args:
            x (torch.Tensor): Input stacked node/item features.
            edge_index (torch.Tensor): [2, E] LongTensor encoding node/item
                connectivity. To exactly mirror transformers, this is fully
                connected for each graph.
            ptr (torch.Tensor): Points to the start/stop of each graph/set
                in the input `:param:x` stacked features. `ptr[i]` points to
                the starting index of graph/set i, and `ptr[i+1]` points to
                the end.
            return_attention_weights (bool, optional): Whether to return the
                attention weights for each node/item when computing the graph/set
                representation as a weighted average. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                If `:param:return_attention_weights=False`, just returns the
                weighted average over all nodes in graph (items in set).
                Shape [batch_size, D]. Otherwise, also returns the attention weights,
                or the scores that computed the weighted average.
        """
        attn_score = self.pooling_attn_layer(
            x, edge_index, return_attention_weights=False
        )

        if self.multiplier != 1.0:
            attn_score = attn_score * self.multiplier

        # shape: [N, 1]
        weights = softmax(attn_score, ptr=ptr, dim=0)

        # x shape: [N, embedding_dim]
        # output shape: [batch_size, embedding_dim]
        # TODO: model bias? I don't think my original impl does
        # main goal is just that the weights are learnable
        weighted_avg = segment(x * weights, ptr, reduce="sum")
        if return_attention_weights:
            return weighted_avg, weights

        return weighted_avg


class FeedForward(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor
        h = self.w1(x)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.w2(h)
        return h


class FixedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_size: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        position = torch.arange(0, max_size, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)
        self.emb: torch.Tensor

    def forward(self, x: torch.Tensor, sizes: Iterable[int]) -> torch.Tensor:
        # get position for each ptn in each genome
        relpos = torch.cat([torch.arange(size) for size in sizes])
        encoding = self.emb[relpos].to(device=x.device)
        return encoding + x


class ResidualMultiheadAttentionConv(MultiheadAttentionConv):
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

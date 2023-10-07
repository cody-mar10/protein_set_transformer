from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
from torch import nn
from torch_geometric.nn import GraphNorm, MessagePassing
from torch_geometric.utils import add_self_loops, segment, softmax

from pst.typing import OptEdgeAttnOutput, OptGraphAttnOutput, OptTensor, PairTensor

from .utils.attention import attention_scale, cat_heads, uncat_heads


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
        self.scale = attention_scale(out_channels)
        self.dropout = dropout
        self._alpha = None

        self.linQ = nn.Linear(in_channels, self.eff_out_channels)
        self.linK = nn.Linear(in_channels, self.eff_out_channels)
        self.linV = nn.Linear(in_channels, self.eff_out_channels)

        self.norm_input = GraphNorm(in_channels)

        if concat:
            # aggr_out will have shape: [-1, H * D]
            self.normO = GraphNorm(self.eff_out_channels)
            self.linO = nn.Linear(
                in_features=self.eff_out_channels, out_features=self.eff_out_channels
            )
        else:
            # aggr_out will have shape: [-1, D]
            self.normO = GraphNorm(self.out_channels)
            self.linO = nn.Linear(
                in_features=self.out_channels, out_features=self.out_channels
            )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.linQ.reset_parameters()
        self.linK.reset_parameters()
        self.linV.reset_parameters()
        self.linO.reset_parameters()
        self.norm_input.reset_parameters()
        self.normO.reset_parameters()

    def uncat_heads(self, x: torch.Tensor) -> torch.Tensor:
        return uncat_heads(x, heads=self.heads)

    def cat_heads(self, x: torch.Tensor) -> torch.Tensor:
        return cat_heads(x)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptEdgeAttnOutput:
        """Forward pass to compute scaled-dot product attention to update each
        node/item representation. This only supports self-attention.

        Args:
            x (torch.Tensor): Stacked node/item feature tensor [N, D]
            edge_index (torch.Tensor): [2, E] LongTensor encoding node/item
                connectivity. To exactly mirror transformers, this is fully
                connected for each graph.
            batch (torch.Tensor): [N] LongTensor encoding the graph/set each node/item
                belongs to.
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
        x = self.norm_input(x, batch)

        # shape [N, H * D] -> [N, H, D]
        query = self.uncat_heads(self.linQ(x))
        key = self.uncat_heads(self.linK(x))
        value = self.uncat_heads(self.linV(x))
        edge_index, _ = add_self_loops(edge_index)

        # propagate order:
        # 1. self.message
        # 2. self.aggregrate
        # 3. self.update
        out = self.propagate(
            edge_index=edge_index,
            initial=x_input,
            query=query,
            key=key,
            value=value,
            batch=batch,
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
        qk = reduce(
            query_i * key_j / self.scale,
            "edges heads dim -> edges heads",
            "sum",
        )

        # shape: [E, H]
        alpha = softmax(qk, index=index, ptr=ptr, num_nodes=size_i, dim=0)

        # store attn weights
        self._push_alpha(alpha)

        # shape: [E, H, D]
        out = value_j * rearrange(alpha, "edges heads -> edges heads 1")
        return out

    def update(
        self, aggr_out: torch.Tensor, initial: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        if self.concat:
            # shape: [N, H, D] -> [N, H * D]
            aggr_out = self.cat_heads(aggr_out)
        else:
            # shape: [N, H, D] -> [N, D]
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
        normed_aggr_out = self.normO(aggr_out, batch)
        normed_aggr_out = F.dropout(
            F.gelu(self.linO(normed_aggr_out)),
            p=self.dropout,
            training=self.training,
        )

        # linO is a 2-layer feedforward network
        aggr_out = aggr_out + normed_aggr_out

        return aggr_out

    # these two methods are convenience methods to pass around the attn weights
    # since the PyG MessagePassing framework doesn't explicitly allow
    # for returning multiple values from the self.propagate method
    def _push_alpha(self, alpha: torch.Tensor):
        self._alpha = alpha.detach()

    def _pop_alpha(self) -> torch.Tensor:
        alpha: torch.Tensor = self._alpha  # type: ignore
        self._alpha = None
        return alpha


# Modeled based off the SAGPooling operation
class MultiheadAttentionPooling(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        heads: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a multihead attention pooling module. This projects the
        input features onto a single dimension, and uses those scores to
        perform a weighted average.

        Args:
            feature_dim (int): Input embedding dimension.
            heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate for the attention calculation.
                Defaults to 0.0.
        """
        dim_per_head, remainder = divmod(feature_dim, heads)
        if remainder != 0:
            raise ValueError(f"{feature_dim=} must be divisible by {heads=}")

        super().__init__()
        # this setup will closely mirror the PoolingMultiheadAttention
        # operation from the SetTransformer paper
        # we can use a variant of scaled dot product attention here
        # instead of relying on the edgewise attention with a graph transformer

        # will project all nodes onto this seed vector
        self.seed = nn.Parameter(torch.empty((feature_dim,)))

        self.dim_per_head = dim_per_head
        self.scale = attention_scale(self.dim_per_head)

        # we don't change the feature dim of any inputs
        # we just let each attention head attend to a subset of the input
        self.linQ = nn.Linear(feature_dim, feature_dim)
        self.linK = nn.Linear(feature_dim, feature_dim)
        self.linV = nn.Linear(feature_dim, feature_dim)
        self.linO = nn.Linear(feature_dim, feature_dim)

        self.normKV = GraphNorm(feature_dim)
        self.normO = GraphNorm(feature_dim)

        self.feature_dim = feature_dim
        self.dropout = dropout
        self.heads = heads

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.seed.unsqueeze(0))

        self.linQ.reset_parameters()
        self.linK.reset_parameters()
        self.linV.reset_parameters()
        self.linO.reset_parameters()
        self.normKV.reset_parameters()
        self.normO.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        ptr: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptGraphAttnOutput:
        """Compute an attention-based score by projecting the input `x`
        onto a single dimension. These scores are then converted to probabilites
        using the softmax function, and then used to perform a weighted
        average along the inputs.

        Args:
            x (torch.Tensor): [N, D] Input stacked node/item features.
            ptr (torch.Tensor): Points to the start/stop of each graph/set
                in the input `:param:x` stacked features. `ptr[i]` points to
                the starting index of graph/set i, and `ptr[i+1]` points to
                the end.
            batch (torch.Tensor): [N] LongTensor encoding the graph/set each node/item
                belongs to.
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
        # all calculations happen at the node level until the final reduction

        # shape: [N, D]
        x_input = x

        # normalize inputs
        K: torch.Tensor = self.normKV(x, batch)
        V = K

        # apply linear weights first
        Q = self.linQ(self.seed)
        K = self.linK(K)
        V = self.linV(V)

        # reshape inputs
        Q = self.uncat_heads(Q)
        K = self.uncat_heads(K)
        V = self.uncat_heads(V)

        # weighted V shape: [N, D]
        weighted_V, attn = self.scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            ptr=ptr,
            return_attention_weights=return_attention_weights,
        )

        # residual connection
        weighted_V = weighted_V + x_input
        norm_weighted_V = self.normO(weighted_V, batch)

        norm_weighted_V = F.dropout(
            F.gelu(self.linO(norm_weighted_V)),
            p=self.dropout,
            training=self.training,
        )

        # residual connection, shape: [N, D]
        res_V = weighted_V + norm_weighted_V

        # shape: [N, D] -> [batch_size, D]
        weighted_graph_avg = segment(res_V, ptr=ptr, reduce="mean")

        return weighted_graph_avg, attn

    def uncat_heads(self, x: torch.Tensor) -> torch.Tensor:
        return uncat_heads(x, heads=self.heads)

    def cat_heads(self, x: torch.Tensor) -> torch.Tensor:
        return cat_heads(x)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        ptr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptGraphAttnOutput:
        # internally this unsqueezes both tensors:
        # [N, H, D, 1] x [N, H, D, 1]^T -> [N, H, D, D]
        # then [N, H, D, D] -> [N, H] by summing over the last 2 dims
        # thus, this represents H scores per node
        attn_weight = einsum(
            Q, K, "heads dim1, nodes heads dim2 -> nodes heads"
        ).div(self.scale)

        # sparsely evaluated softmax that normalizes each attn head per graph
        # thus each attn head is weighting the importance of each node in the graph
        attn_weight = softmax(attn_weight, ptr=ptr, dim=0)

        # we can't do the typical matmul again since the weights
        # correspond to each node per graph in a stacked batch
        weighted_V = V * attn_weight.unsqueeze(-1)

        # shape: [N, H, D'] -> [N, D]
        weighted_V = self.cat_heads(weighted_V)

        if return_attention_weights:
            return weighted_V, attn_weight
        return weighted_V, None


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        **actv_kwargs,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, in_dim)
        self.activation = activation(**actv_kwargs)
        self.dropout = dropout
        self.w2 = nn.Linear(in_dim, out_dim)

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

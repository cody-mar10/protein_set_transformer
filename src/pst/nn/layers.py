from typing import Optional

import torch
from einops import einsum, rearrange, reduce
from torch import nn
from torch_geometric.nn import MLP, GraphNorm, MessagePassing
from torch_geometric.utils import add_self_loops, segment, softmax

from pst.nn.utils.attention import AttentionMixin
from pst.nn.utils.norm import NormMixin
from pst.typing import EdgeAttnOutput, GraphAttnOutput, OptTensor


class MultiheadAttentionConv(MessagePassing, AttentionMixin, NormMixin):
    """Implement the SetTransformer attention layer using a graph-level
    implementation. Graph implementations are useful when the number of items
    greatly varies among sets, resulting in substantial memory and computation
    savings.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        dropout: float = 0.0,
        add_self_loops: bool = False,
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
            dropout (float, optional): Dropout fraction for the final
                linear layer. Defaults to 0.0.
            add_self_loops (bool, optional): Whether to add self-loops to the
                input graph. Defaults to False.
            **kwargs: Additional arguments for the `MessagePassing` base class
        """
        # this setup mirrors exactly how attention would be computed normally
        MessagePassing.__init__(self, aggr="add", node_dim=0, **kwargs)
        AttentionMixin.__init__(self, in_dim=in_channels, heads=heads)

        if out_channels > in_channels:
            raise ValueError(
                f"out_channels {out_channels} must be <= in_channels {in_channels}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._alpha = None
        self.add_self_loops = add_self_loops

        # scaled dot product attn does not change feature dim
        self.linQ, self.linK, self.linV = self.init_weight_layers(in_channels)
        self.norm_input = GraphNorm(in_channels)

        self.normO = GraphNorm(in_channels)
        self.linO = MLP(
            channel_list=[in_channels, in_channels, out_channels],
            norm=None,
            dropout=dropout,
            plain_last=True,
            act="gelu",
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_mask: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> EdgeAttnOutput:
        """Forward pass to compute scaled-dot product attention to update each
        node/item representation. This only supports self-attention.

        Args:
            x (torch.Tensor): Stacked node/item feature tensor [N, D]
            edge_index (torch.Tensor): [2, E] LongTensor encoding node/item
                connectivity. To exactly mirror transformers, this is fully
                connected for each graph.
            batch (torch.Tensor): [N] LongTensor encoding the graph/set each node/item
                belongs to.
            node_mask (OptTensor, optional): Masking tensor to apply to the attention weights.
                `True` values are masked out. Internally, this just removes edges that contain
                masked nodes to they don't actually contribute to the final representations.
                Defaults to None (no masking).
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
        x = self.normalize(x, self.norm_input, batch)

        # shape [N, H * D] -> [N, H, D]
        query = self.uncat_heads(self.linQ(x))
        key = self.uncat_heads(self.linK(x))
        value = self.uncat_heads(self.linV(x))

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index)

        if node_mask is not None:
            # if either node is masked, the edge is masked
            # True = masked out
            edge_mask = node_mask[edge_index[0]] | node_mask[edge_index[1]]

            # then just remove any edges that are masked, so attn is not computed
            # along those edges
            edge_index = edge_index[:, ~edge_mask]

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
            return EdgeAttnOutput(out=out, edge_index=edge_index, attn=alpha)

        return EdgeAttnOutput(out=out, edge_index=None, attn=None)

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
            query_i * key_j / self.attention_scale,
            "edges heads dim -> edges heads",
            "sum",
        )

        alpha = softmax(qk, index=index, ptr=ptr, num_nodes=size_i, dim=0)

        # store attn weights
        self._push_alpha(alpha)

        # shape: [E, H, D]
        out = value_j * rearrange(alpha, "edges heads -> edges heads 1")
        return out

    def update(
        self, aggr_out: torch.Tensor, initial: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        # shape [N, H, D] -> [N, H * D] where H * D is the original feature dim
        aggr_out = self.cat_heads(aggr_out)

        # residuals before normalization, ie pre-norm
        # these should always be the same size
        aggr_out = self.residual_connection(aggr_out, initial)
        normed_aggr_out: torch.Tensor = self.normalize(aggr_out, self.normO, batch)

        # linO is a 2-layer feed forward network
        # shape: [N, D] -> [N, D']
        normed_aggr_out = self.linO(normed_aggr_out)

        aggr_out = self.residual_connection(normed_aggr_out, aggr_out)

        return aggr_out

    def residual_connection(
        self, output: torch.Tensor, initial: torch.Tensor
    ) -> torch.Tensor:
        out_dim = output.size(-1)
        initial_dim = initial.size(-1)

        if out_dim == initial_dim:
            return output + initial

        # just over nodes to add
        # alternative would be to concatenate 0s to the end as a pad
        # NOTE: this does not allow the output to be larger than the input
        return output + reduce(initial, "nodes dim -> nodes 1", "mean")

    # these two methods are convenience methods to pass around the attn weights
    # since the PyG MessagePassing framework doesn't explicitly allow
    # for returning multiple values from the self.propagate method
    def _push_alpha(self, alpha: torch.Tensor):
        self._alpha = alpha.detach()

    def _pop_alpha(self) -> torch.Tensor:
        alpha: torch.Tensor = self._alpha  # type: ignore
        self._alpha = None
        return alpha


class MultiheadAttentionPooling(nn.Module, AttentionMixin, NormMixin):
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
        nn.Module.__init__(self)
        AttentionMixin.__init__(self, in_dim=feature_dim, heads=heads)
        # this setup will closely mirror the PoolingMultiheadAttention
        # operation from the SetTransformer paper
        # we can use a variant of scaled dot product attention here
        # instead of relying on the edgewise attention with a graph transformer

        # will project all nodes onto this seed vector
        self.seed = nn.Parameter(torch.empty((1, feature_dim)))

        # we don't change the feature dim of any inputs
        # we just let each attention head attend to a subset of the input
        self.linQ, self.linK, self.linV = self.init_weight_layers(feature_dim)

        self.linO = MLP(
            in_channels=feature_dim,
            hidden_channels=feature_dim,
            out_channels=feature_dim,
            num_layers=2,
            act="gelu",
            dropout=dropout,
            norm=None,
            plain_last=True,
        )

        self.normKV = GraphNorm(feature_dim)
        self.normO = GraphNorm(feature_dim)

        self.feature_dim = feature_dim

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.seed)

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
    ) -> GraphAttnOutput:
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
        K: torch.Tensor = self.normalize(x, self.normKV, batch)
        V: torch.Tensor = K

        # apply linear weights first
        Q: torch.Tensor = self.linQ(self.seed)
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
        norm_weighted_V: torch.Tensor = self.normalize(weighted_V, self.normO, batch)
        norm_weighted_V = self.linO(norm_weighted_V)

        # residual connection, shape: [N, D]
        res_V = weighted_V + norm_weighted_V

        # shape: [N, D] -> [batch_size, D]
        # TODO: this actually should be sum....
        # actually dont think so since there are several steps between this and the weighted_V..
        weighted_graph_avg = segment(res_V, ptr=ptr, reduce="mean")

        return GraphAttnOutput(out=weighted_graph_avg, attn=attn)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        ptr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> GraphAttnOutput:
        # internally this unsqueezes both tensors:
        # [1, H, D, 1] x [N, H, D, 1]^T -> [N, H, D, D]
        # then [N, H, D, D] -> [N, H] by summing over the last 2 dims
        # thus, this represents H scores per node
        attn_weight = einsum(
            Q, K, "unit heads dim1, nodes heads dim2 -> nodes heads"
        ).div(self.attention_scale)

        # sparsely evaluated softmax that normalizes each attn head per graph
        # thus each attn head is weighting the importance of each node in the graph
        attn_weight = softmax(attn_weight, ptr=ptr, dim=0)

        # we can't do the typical matmul again since the weights
        # correspond to each node per graph in a stacked batch
        weighted_V = V * attn_weight.unsqueeze(-1)

        # shape: [N, H, D'] -> [N, D]
        weighted_V = self.cat_heads(weighted_V)

        if return_attention_weights:
            return GraphAttnOutput(out=weighted_V, attn=attn_weight)
        return GraphAttnOutput(out=weighted_V, attn=None)


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

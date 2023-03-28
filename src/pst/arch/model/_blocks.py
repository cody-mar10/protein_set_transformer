from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pst.utils.mask import compute_row_mask, row_mask_to_attn_mask


@dataclass
class AttentionSchema:
    repr: torch.Tensor
    weights: Optional[torch.Tensor]


class MultiheadAttention(nn.Module):
    """Multiheaded self-attention block for set transformers"""

    def __init__(
        self,
        qdim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        norm: bool = True,
    ) -> None:
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        kdim = kdim or qdim
        vdim = vdim or qdim

        for dim in (qdim, kdim, vdim):
            if dim % num_heads != 0:
                raise ValueError(
                    "The embedding dimensions of the query/key/value must be divisible by the number of attention heads."
                )

        # learnable weights for Q/K/V
        self.Wq = nn.Linear(qdim, vdim, bias=bias)
        self.Wk = nn.Linear(kdim, vdim, bias=bias)
        self.Wv = nn.Linear(kdim, vdim, bias=bias)
        self.Wo = nn.Linear(vdim, vdim, bias=bias)
        self.layernorm = nn.LayerNorm(vdim) if norm else None

    def to_multiheaded(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, n_seqs, feat_dim = X.size()
        sub_dim = feat_dim // self.num_heads
        # reshape input to [batch_size * num_heads, n_seqs, sub_dim]
        # basically divide the feat_dim up into each head
        multiheaded_view = (
            X.reshape(batch_size, n_seqs, self.num_heads, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, n_seqs, sub_dim)
        )
        return multiheaded_view

    def from_multiheaded(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, n_seqs, feat_dim = X.size()
        batch_size //= self.num_heads
        out_dim = feat_dim * self.num_heads
        concatenated_view = (
            X.reshape(batch_size, self.num_heads, n_seqs, feat_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, n_seqs, out_dim)
        )
        return concatenated_view

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_weights: bool = True,
    ) -> AttentionSchema:
        scale = 1 / math.sqrt(Q.size(-1))
        if attn_mask is None:
            # attending to any position is allowed
            # likely never reached if calling from self.forward
            batch_size = Q.size(0)
            inner = Q.size(-2)
            outer = K.size(-2)
            attn_mask = torch.ones(batch_size, inner, outer, dtype=torch.bool).to(
                device=Q.device
            )

        attn_mask = torch.where(attn_mask, float("-inf"), 0.0)

        attn_weight = torch.softmax(
            (Q @ K.transpose(-2, -1) * scale) + attn_mask, dim=-1
        )
        # since each row for a genome is a ptn and entire rows are padded with 0s
        # the padded rows will all be nan after softmax since
        # they sum to nan still
        # replacing them with 0s shouldnt? affect calcs
        # since matmul with a 0 vector is 0
        # idk if this is needed since ISAB won't have 0 rows

        # SWITCHING TO NOT USE INPLACE OPS
        # attn_weight.nan_to_num_(nan=0.0)
        attn_weight = attn_weight.nan_to_num(nan=0.0)
        attn_weight = torch.dropout(attn_weight, self.dropout, self.training)
        attn = attn_weight @ V
        return AttentionSchema(
            repr=attn, weights=attn_weight if return_weights else None
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        return_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> AttentionSchema:
        # row masks computed pre linear layers to know which rows were all 0s aka padded
        # only need masks for Q and K due to attention calc
        Q_row_mask = compute_row_mask(Q)

        # default is self-attention
        if K is None:
            K = Q
            K_row_mask = Q_row_mask
        else:
            K_row_mask = compute_row_mask(K)

        if attn_mask is None:
            # compute attn_mask solely based on rows that are/aren't padded rows
            # also repeat to match multiheaded attn dimensions
            attn_mask = row_mask_to_attn_mask(
                Q_row_mask=Q_row_mask,
                K_row_mask=K_row_mask,
            ).repeat_interleave(repeats=self.num_heads, dim=0)

        if V is None:
            V = K

        # apply learnable weights
        Q = F.relu(self.Wq(Q))
        K = F.relu(self.Wk(K))
        V = F.relu(self.Wv(V))

        # reshape for multiheaded attn
        Q = self.to_multiheaded(Q)
        K = self.to_multiheaded(K)
        V = self.to_multiheaded(V)

        # attn calculation
        output = self.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, return_weights=return_weights
        )

        # SWITCHING TO NOT USE INPLACE OPS
        # residual connection
        output.repr = output.repr + Q

        # concatenate attn heads
        output.repr = self.from_multiheaded(output.repr)

        if self.layernorm is not None:
            output.repr = self.layernorm(output.repr)

        output.repr = F.relu(output.repr)
        return output


class SetAttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        norm: bool = True,
    ) -> None:
        super(SetAttentionBlock, self).__init__()
        self.mab = MultiheadAttention(
            qdim=in_dim,
            vdim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            norm=norm,
        )

    def forward(
        self,
        X: torch.Tensor,
        return_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> AttentionSchema:
        return self.mab(Q=X, return_weights=return_weights, attn_mask=attn_mask)


class InducedSetAttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        num_indices: int,
        dropout: float = 0.0,
        bias: bool = True,
        norm: bool = True,
    ) -> None:
        super(InducedSetAttentionBlock, self).__init__()
        self.I = nn.Parameter(torch.empty((1, num_indices, out_dim)))
        nn.init.xavier_normal_(self.I)

        # project to lower dimensional space defined by num_indices
        self.iproj_mab = MultiheadAttention(
            qdim=out_dim,
            kdim=in_dim,
            vdim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            norm=norm,
        )
        self.mab = MultiheadAttention(
            qdim=in_dim,
            kdim=out_dim,
            vdim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            norm=norm,
        )

    def forward(
        self,
        X: torch.Tensor,
        return_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> AttentionSchema:
        # project X to lower dimension space first
        H: torch.Tensor = self.iproj_mab(
            Q=self.I.repeat(X.size(0), 1, 1),
            K=X,
            return_weights=False,
            attn_mask=attn_mask,
        ).repr
        return self.mab(Q=X, K=H, return_weights=return_weights, attn_mask=attn_mask)


class PooledMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_seeds: int,
        dropout: float = 0.0,
        bias: bool = True,
        norm: bool = True,
    ):
        super(PooledMultiheadAttention, self).__init__()
        self.S = nn.Parameter(torch.empty((1, num_seeds, embed_dim)))
        nn.init.xavier_uniform_(self.S)

        self.mab = MultiheadAttention(
            qdim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            norm=norm,
        )

    def forward(
        self,
        X: torch.Tensor,
        return_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> AttentionSchema:
        return self.mab(
            Q=self.S.repeat(X.size(0), 1, 1),
            K=X,
            return_weights=return_weights,
            attn_mask=attn_mask,
        )

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._norm import SetNorm
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
        normalize_Q: bool = True,
        # from setnorm impl, unclear what they are for
        sample_size: int = 1000,
        v_norm_samples: int = 32,
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
        self.normalize_Q = normalize_Q
        self.normQ = SetNorm(
            feature_dim=qdim,
            normalized_shape=(sample_size, qdim),
            elementwise_affine=False,
        )
        self.normK = SetNorm(
            feature_dim=kdim,
            normalized_shape=(v_norm_samples, kdim),
            elementwise_affine=False,
        )
        self.normVO = SetNorm(
            feature_dim=vdim,
            normalized_shape=(sample_size, vdim),
            elementwise_affine=False,
        )

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

    def from_multiheaded_weights(self, X: torch.Tensor) -> torch.Tensor:
        eff_batch_size, m_seqs, n_seqs = X.size()
        batch_size = eff_batch_size // self.num_heads
        return X.reshape(batch_size, self.num_heads, m_seqs, n_seqs)

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

        # cant use float("-inf") as normal since padded all 0 rows
        # will turn into nan rows, which causes gradient and backprop problems
        attn_fill = torch.where(attn_mask, -5e4, 0.0)

        # also use attn_mask to zero out padded rows/ptns/items
        # above calc in softmax ignores real-padded item attn
        # below calc ignores padded-padded item attn
        attn_weight = (
            torch.softmax((Q @ K.transpose(-2, -1) * scale) + attn_fill, dim=-1)
            * ~attn_mask
        )

        # TODO: SetTransformer and ESM papers don't use dropout here since this reduces capacity?
        # attn_weight = torch.dropout(attn_weight, self.dropout, self.training)
        attn = attn_weight @ V  # m x n
        return AttentionSchema(
            repr=attn,
            # weights shape: [b, h, m, n]
            # where b = batch size, h = attn heads, m = Q set size, n = K set size
            # NOTE: to combine weights across attn heads, need to do this across dim=1
            weights=self.from_multiheaded_weights(attn_weight)
            if return_weights
            else None,
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        return_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> AttentionSchema:
        # keep this for "clean path" residual connection
        # see SetNorm paper
        Q_input = Q

        # row masks computed pre linear layers to know which rows were all 0s aka padded
        # only need masks for Q and K due to attention calc
        Q_row_mask = compute_row_mask(Q, unsqueeze=True)

        # default is self-attention
        if K is None:
            K = Q
            K_row_mask = Q_row_mask
        else:
            K_row_mask = compute_row_mask(K, unsqueeze=True)

        if V is None:
            V = K
            V_row_mask = K_row_mask
        else:
            V_row_mask = compute_row_mask(V, unsqueeze=False)

        # pre-attention normalization
        if self.normalize_Q:
            Q = self.normQ(Q, row_mask=Q_row_mask)

        K = self.normK(K, row_mask=K_row_mask)

        if attn_mask is None:
            # compute attn_mask solely based on rows that are/aren't padded rows
            # also repeat to match multiheaded attn dimensions
            attn_mask = row_mask_to_attn_mask(
                Q_row_mask=Q_row_mask,
                K_row_mask=K_row_mask,
            ).repeat_interleave(repeats=self.num_heads, dim=0)

        # apply learnable weights and mask padded rows
        Q = self.Wq(Q) * Q_row_mask
        K = self.Wk(K) * K_row_mask
        V = self.Wv(V) * K_row_mask

        # attn calculation with inputs reshaped for multiheaded attn
        output = self.scaled_dot_product_attention(
            Q=self.to_multiheaded(Q),
            K=self.to_multiheaded(K),
            V=self.to_multiheaded(V),
            attn_mask=attn_mask,
            return_weights=return_weights,
        )

        # concatenate attn heads
        attn_output = self.from_multiheaded(output.repr)

        # residual connection
        # original transformer applies dropout to attn_output here
        attn_output = attn_output + Q_input
        normed_attn_output = self.normVO(attn_output, V_row_mask)

        # output feed-forward and another set norm with residual connections
        # using pre-transformation normalization
        normed_attn_output = self.Wo(normed_attn_output)
        output.repr = attn_output + torch.dropout(
            F.relu(normed_attn_output), self.dropout, self.training
        )
        return output


class SetAttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        normalize_Q: bool = True,
        sample_size: int = 1,
    ) -> None:
        super(SetAttentionBlock, self).__init__()
        self.mab = MultiheadAttention(
            qdim=in_dim,
            vdim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            normalize_Q=normalize_Q,
            sample_size=sample_size,
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
        normalize_Q: bool = True,
        sample_size: int = 100,
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
            normalize_Q=False,
            sample_size=num_indices,
            v_norm_samples=sample_size,
        )
        self.mab = MultiheadAttention(
            qdim=in_dim,
            kdim=out_dim,
            vdim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            normalize_Q=normalize_Q,
            sample_size=sample_size,
            v_norm_samples=num_indices,
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
        normalize_Q: bool = True,
    ):
        super(PooledMultiheadAttention, self).__init__()
        self.S = nn.Parameter(torch.empty((1, num_seeds, embed_dim)))
        nn.init.xavier_uniform_(self.S)

        self.mab = MultiheadAttention(
            qdim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            normalize_Q=normalize_Q,
            sample_size=1,
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

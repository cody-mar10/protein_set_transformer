from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pst.utils.mask import compute_row_mask


class SetNorm(nn.LayerNorm):
    # From: https://arxiv.org/pdf/2206.11925.pdf
    #       https://github.com/rajesh-lab/deep_permutation_invariant/blob/main/models/norms.py
    def __init__(self, feature_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._weight = nn.Parameter(torch.empty(feature_dim))
        self._bias = nn.Parameter(torch.empty(feature_dim))
        nn.init.constant_(self._weight, 1.0)
        nn.init.constant_(self._bias, 0.0)

    def forward(
        self, X: torch.Tensor, row_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if row_mask is None:
            row_mask = compute_row_mask(X)

        # add all items from set and then divide by number of true elements
        # true elements come from real rows, ie not row padded rows
        n_true_elements = row_mask.sum(dim=-1) * X.size(-1)

        # 2 unsqueezes needed to broadcast correctly
        per_set_global_mean = (
            X.sum(dim=(-2, -1)).div(n_true_elements).unsqueeze(-1).unsqueeze(-1)
        )
        per_set_global_std = (
            torch.square(X - per_set_global_mean)
            .sum(dim=(-2, -1))
            .div(n_true_elements)
            .sqrt()
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        # always keep padded rows zeroed out
        Xnorm = (
            (X - per_set_global_mean)
            / (per_set_global_std + self.eps)
            * row_mask.unsqueeze(-1)
        )
        return F.linear(Xnorm, torch.diag_embed(self._weight), self._bias)

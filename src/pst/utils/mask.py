from __future__ import annotations

from typing import Optional

import torch


def compute_row_mask(X: torch.Tensor) -> torch.Tensor:
    """Compute a row mask for input row-padded tensor.

    Args:
        X (torch.Tensor): row-padded tensor with shape [b, m, d]
            b: batch size
            m: number of items
            d: feature dimension of item

    Returns:
        torch.Tensor: boolean tensor of shape [b, m, 1] that indicates which rows are
            real. True = real row. False = padded row of all 0s.
    """
    return torch.unsqueeze(X.abs().sum(-1) != 0.0, dim=-1)


def row_mask_to_attn_mask(
    Q_row_mask: torch.Tensor, K_row_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # defaults to self-attn
    if K_row_mask is None:
        K_row_mask = Q_row_mask

    # row masks have True for real rows and False for padded (all 0s) rows
    # but attn masks have True for positions where attn is not allowed
    # and False for allowed positions
    # so need to invert
    attn_mask = (
        torch.matmul(Q_row_mask.float(), K_row_mask.float().transpose(-2, -1))
        .bool()
        .logical_not()
    )
    return attn_mask

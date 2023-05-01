from __future__ import annotations

from typing import Optional

import torch
from pst.utils.mask import compute_row_mask, pairwise_row_mask


def _reshape_flat_pairwise_comparison_tensor(
    flat_pairwise_comp: torch.Tensor, batch_size: int, n_items: int
) -> torch.Tensor:
    reshaped = flat_pairwise_comp.reshape(
        batch_size, n_items, batch_size, n_items
    ).permute(2, 0, 3, 1)
    return reshaped


# TODO: may need to add a custom CUDA/triton kernel to prevent high mem usage
# all dist mats are [n, n], so we could technically just have a [(b^2-b/2), n, n]
# total dist mat, and then just convert (i, j) genome indices to a specific idx
# in this distance matrix
def batch_chamfer_distance(
    X: torch.Tensor,
    row_mask: Optional[torch.Tensor] = None,
    average: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate Chamfer distance between all pairs of sets within a batch. The Chamfer
    distance is symmetric such that `d(X1, X2) == d(X2, X1)`.

    Note: The original implementation uses squared Euclidean distance, but this uses
    Euclidean distance, which should have the same overall interpretation.

    Args:
        X (torch.Tensor): a batched point-set tensor, shape [b, n, d]
        row_mask (Optional[torch.Tensor], optional): The row mask for X that denotes
            which rows correspond to real rows vs padded rows. Defaults to None.
        average (bool, optional): Choose to normalize the distances by the number of items
            in each individual set. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:

            - [b, b] Chamfer distance tensor between all sets in the batched tensor
            - [b, b, n] minimum distance index tensor. For an index (i, j, k), this
                corresponds to the closest item in set j to item k in set i.
    """
    # flatten X to [b*n, d] matrix, so all ptns are along the 1st dim
    X_flat = X.flatten(start_dim=0, end_dim=-2)

    # [b*n, b*n]
    # this matrix has all ptn comparisons, and blocks of size [n, n] correspond to specific
    # genome-genome comparisons
    all_distances: torch.Tensor = torch.cdist(X_flat, X_flat, p=2).fill_diagonal_(0.0)
    batch_size, n_items, _ = X.size()

    # which means if we just reshape and permute this matrix, we can convert it to a 4d tensor
    # where the index (i, j, m, n) corresponds to the distance between
    # ptn m from genome i and ptn n from genome j
    # [b, b, n, n] <- index which genome idx to get the dist mat for each genome comparison
    # NOTE: potentially could be faster to call .contiguous() but tests on colab indicated otherwise
    # plus that would require a full copy, which is inefficient since we're just calling .min
    all_distances = _reshape_flat_pairwise_comparison_tensor(
        all_distances, batch_size, n_items
    )

    if row_mask is None:
        row_mask = compute_row_mask(X)

    flat_row_mask = row_mask.flatten()
    # mask same shape as all_distances but masks out pairwise distance between any padded row
    full_mask = _reshape_flat_pairwise_comparison_tensor(
        pairwise_row_mask(flat_row_mask, flat_row_mask), batch_size, n_items
    )
    filled_mask = torch.where(full_mask, 0.0, float("inf"))

    min_dists: torch.Tensor
    item_flow_indices: torch.Tensor
    # for all genome-genome comparisons find the min dist for all ptns but cannot be between padded rows
    min_dists, item_flow_indices = torch.min(all_distances + filled_mask, dim=-1)

    # convert inf dist with padded rows to 0.0 so they don't count towards distance
    min_dists[torch.where(min_dists == float("inf"))] = 0.0

    item_distance = min_dists.sum(dim=-1)
    if average:
        item_distance /= row_mask.sum(-1).unsqueeze(-1)

    # consider distance in both directions for each genome-genome interaction
    # this is symmetric
    chamfer_dist = item_distance + item_distance.transpose(-2, -1)
    return chamfer_dist, item_flow_indices

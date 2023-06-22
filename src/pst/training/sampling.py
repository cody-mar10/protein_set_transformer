from __future__ import annotations

from typing import Optional

import torch

from .distance import pairwise_chamfer_distance
from ._types import FlowType


def positive_sampling(
    setdist: torch.Tensor, mask_diagonal: bool = True
) -> torch.Tensor:
    """Perform positive sampling using a set-wise distance metric like Chamfer distance.

    It is recommended that the set-wise distance be calculated BEFORE a model forward
    pass since this will be comparing data in the original dimension compared to the
    embedding dimension. The positive pairs in the embedding dimension will be heavily
    dependent on the model and how it is being trained and how it was initialized.

    Args:
        setdist (torch.Tensor)
        mask_diagonal (bool): use if the diagonal needs to be masked since that
            is likely a zero vector. In other words, don't choose self as the
            positive index. Defaults to True.

    Returns:
        torch.Tensor: tensor with the indices of the positive set
    """
    # choose smallest k distances
    # the diagonal is likely all 0s, so that will be the smallest values per row
    k = 2 if mask_diagonal else 1
    pos_sample_idx: torch.Tensor = setdist.topk(k, dim=-1, largest=False)[1][:, -1]
    return pos_sample_idx


def negative_sampling(
    setwise_dist: torch.Tensor,
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    *,
    pos_idx: Optional[torch.Tensor] = None,
    scale: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if Y is None:
        # this chooses the neg sample from the real dataset
        # otherwise, Y is the augmented data, and this will choose from Y
        Y = X
        if pos_idx is None:
            raise ValueError("pos_idx must be supplied if a second tensor Y is not")
    else:
        # pos_idx is just aligned already in the case of sampling from point swapped augmented data
        pos_idx = torch.arange(Y.size(0))

    # in case of sampling from real dataset, diag should be 0 due to self-comparison
    # so don't allow self to be chosen as neg sample
    # for pnt swap, generally diagonal should be the smallest, so chose another
    embed_dist = torch.cdist(X, Y)

    # don't allow points closer than the pos sample be chosen
    # this also has the effect of removing self-comparisons
    pos_dists = embed_dist.gather(dim=-1, index=pos_idx.unsqueeze(1))  # type: ignore

    gt_pos = embed_dist > pos_dists
    has_neg_choices = torch.sum(gt_pos, dim=-1).bool()
    dist_from_pos = embed_dist - pos_dists

    if has_neg_choices.all():
        # neg example is the one that is the next farther after
        # dist = 0.0 is pos_idx, dist < 0.0 is closer than pos,
        # but we want just farther than pos
        neg_idx: torch.Tensor = torch.where(
            dist_from_pos <= 0.0, torch.inf, dist_from_pos
        ).min(dim=-1)[1]
    else:
        raise RuntimeError("can't find negative")

        # TODO: check if neg_idx == pos_idx due to rounding errors
    # else:
    #     # TODO: the above can cause a problem if the positive sample is somehow
    #     # farther away than the negative samples in the embedding space,
    #     # which may be common at the beginning of modeling
    #     # next best thing is to choose the 1st index smaller than the positive example
    #     # can just do:

    #     abs_dist_from_pos = dist_from_pos.abs()
    #     abs_dist_from_pos[
    #         torch.where(torch.isclose(dist_from_pos, dist_from_pos.new_zeros(1)))
    #     ] = torch.inf
    #     # the closest negative will have the smallest nonzero distance
    #     idx_closest_to_pos = dist_from_pos.min(dim=-1)[1]

    # calc negative sample weight
    neg_setwise_dists = setwise_dist.gather(
        dim=-1, index=neg_idx.unsqueeze(1)
    ).squeeze()
    setwise_std = setwise_dist.std()
    denom = 2 * (scale * setwise_std) ** 2
    neg_weight = torch.exp(-neg_setwise_dists / denom)

    return neg_idx, neg_weight


class TripletSetSampler:
    """PointSet sampler for data augmentation during trainer. Samples positive examples
    from the original data vector space and samples negative examples from the
    embedding space of the model. This allows for online triplet sampling.
    """

    pass


def point_swap_sampling(
    batch: torch.Tensor,
    ptr: torch.Tensor,
    pos_idx: torch.Tensor,
    item_flow: FlowType,
    sample_rate: float,
) -> torch.Tensor:
    """Perform point swap sampling data augmentation.

    Args:
        batch (torch.Tensor): stacked batch, shape: [total_items, feat_dim]
        ptr (torch.Tensor): data batch ptr to start/stop for each set in
            batch, shape: [batch_size + 1]
        pos_idx (torch.Tensor): at position i, holds the closest set j
            index to set i, shape: [batch_size]
        item_flow (FlowType): dict mapping pairs of indices (i,j) to the
            closest item index for each item m in set i to the items n
            in set j
        sample_rate (float): rate of point swapping, (0.0, 1.0)

    Returns:
        torch.Tensor: _description_
    """
    if sample_rate <= 0.0 or sample_rate >= 1.0:
        raise ValueError(f"Provided {sample_rate=} must be in the range (0.0, 1.0)")

    # build point swap index that maps k-th item in batch to its l-th neighbor
    # as defined by randomly sampling among the available flow
    # for set i and set j comparisons
    point_swap_indices: list[torch.Tensor] = list()
    for i, j in enumerate(pos_idx):
        j = int(j)

        # convert relative item indices to global item indices
        i_offset = ptr[i]
        j_offset = ptr[j]

        pos_item_idx = item_flow[i, j] + j_offset
        n_items = pos_item_idx.shape[0]
        anchor_idx = torch.arange(n_items) + i_offset

        # sample from uniform [0, 1) distribution
        choices = torch.rand(n_items) >= sample_rate
        point_swap_idx = torch.where(choices, pos_item_idx, anchor_idx)
        point_swap_indices.append(point_swap_idx)

    point_swap_index = torch.cat(point_swap_indices)
    augmented_batch = batch[point_swap_index]
    return augmented_batch


def heuristic_augmented_negative_sampling(
    X_anchor: torch.Tensor,
    X_aug: torch.Tensor,
    y_aug: torch.Tensor,
    neg_idx: torch.Tensor,
    ptr: torch.Tensor,
    scale: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    chamfer_distances: list[float] = list()
    for i, j in enumerate(neg_idx):
        j = int(j)

        start_i = ptr[i]
        stop_i = ptr[i + 1]
        start_j = ptr[j]
        stop_j = ptr[j]

        set_i = X_anchor[start_i:stop_i]
        set_j = X_aug[start_j:stop_j]

        chamfer_dist = pairwise_chamfer_distance(set_i, set_j)
        chamfer_distances.append(chamfer_dist)

    aug_neg_dist = torch.tensor(chamfer_distances, device=X_anchor.device)
    ### AUG NEG WEIGHTS ###
    # TODO: using std here may make the weights really small since there are fewer distances
    # gonna use mean for now
    std = aug_neg_dist.std()
    denom = 2 * (scale * std) ** 2
    aug_neg_weights = torch.exp(-aug_neg_dist / denom)

    return y_aug[neg_idx], aug_neg_weights

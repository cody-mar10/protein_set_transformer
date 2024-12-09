from typing import Optional

import torch
from einops import rearrange, reduce

from pst.nn.utils.distance import pairwise_chamfer_distance
from pst.typing import NO_NEGATIVES_MODES, OptTensor, PairTensor


def positive_sampling(pairwise_dist: torch.Tensor, clone: bool = False) -> torch.Tensor:
    """Perform positive sampling using a pairwise distance metric.

    It is recommended that the pairwise distance be calculated BEFORE a model forward
    pass since this will be comparing data in the original dimension compared to the
    embedding dimension. The positive pairs in the embedding dimension will be heavily
    dependent on the model and how it is being trained and how it was initialized.

    For triplet loss, negative examples can be sampled in the model's embedding space. Thus,
    the distance to positives is not relevant, as only the indices of the positive examples
    are needed to select negative examples.

    Args:
        pairwise_dist (torch.Tensor): pairwise distance tensor of shape [N, N]
        clone (bool, optional): whether to clone the input tensor. This is needed if the
            original unedited tensor is needed since this method will mask the diagonal to
            prevent selecting the same point as the positive. Defaults to False.

    Returns:
        torch.Tensor: index tensor with the indices of the positive set, shape [N]
    """
    if clone:
        pairwise_dist = pairwise_dist.clone()

    masked = pairwise_dist.fill_diagonal_(torch.inf)
    return masked.argmin(dim=-1)


def distance_from_index(
    pairwise_dist: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """From a pairwise distance tensor, get the distances from a given index where the position
    of along the first dimension corresponds to the item in the pairwise distance tensor at the
    same position. The index values correspond to the specific columns in the pairwise distance.

    Args:
        pairwise_dist (torch.Tensor): pairwise distance tensor of shape [N, N]
        index (torch.Tensor): index tensor of shape [N], the index value j in position i
            corresponds to position ij in the pairwise distance tensor

    Returns:
        torch.Tensor: distance tensor of shape [N]
    """
    if index.ndim == 1:
        index = rearrange(index, "batch -> batch 1")
    return pairwise_dist.gather(dim=-1, index=index).squeeze()


def _semi_hard_negative_sampling(
    dist_from_pos: torch.Tensor, pos_idx: torch.Tensor, batch_size: Optional[int] = None
) -> torch.Tensor:
    if batch_size is None:
        batch_size = pos_idx.size(0)

    # mask positive idx from being selected -- avoids comparisons with 0
    mask = dist_from_pos.new_full(size=(batch_size, 1), fill_value=torch.inf)
    dist_from_pos = dist_from_pos.scatter(dim=-1, index=pos_idx, src=mask)
    # mask closer than positive idx
    # this is only for when there are actually are negative choices
    # that are greater than the positive idx
    # in other cases when looking for negatives that are closer than the positive
    # just take the absolute value of the dist_from_pos tensor first
    dist_from_pos = torch.where(
        dist_from_pos <= 0.0,
        torch.inf,
        dist_from_pos,
    )
    neg_idx = dist_from_pos.argmin(dim=-1)
    return neg_idx


@torch.no_grad()
def negative_sampling(
    input_space_pairwise_dist: torch.Tensor,
    output_embed_X: torch.Tensor,
    output_embed_Y: OptTensor = None,
    input_space_dist_std: torch.Tensor | float | None = None,
    *,
    pos_idx: OptTensor = None,
    scale: float = 7.0,
    no_negatives_mode: NO_NEGATIVES_MODES = "closest_to_positive",
) -> PairTensor:
    """Sample negative examples for triplet loss using a semi-hard negative sampling strategy
    in which negative examples are chosen from the model's output embedding space. This
    sampling strategy requires a positive sample to be chosen first, which can be done using
    the `positive_sampling` function or from known relationships as with data augmentation.

    Args:
        input_space_pairwise_dist (torch.Tensor): Pairwise distance tensor of the INPUT
            embeddings, shape: [N, N]. This should be the same distance tensor used for
            the positive sampling.
        output_embed_X (torch.Tensor): output embeddings of the model, shape: [N, D]
        Y (OptTensor, optional): output embeddings of the model. Defaults to None.
            This is only used for SETWISE/GRAPHWISE negative sampling when the negative samples
            are chosen from different genomes, such as with Point Swap Augmentation.
        input_space_dist_std (torch.Tensor | float | None, optional): standard deviation of
            the input_space_pairwise_dist tensor. If not provided, it will be calculated from
            the input tensor. It is recommended to pre-compute this if positive sampling is
            needed since the `positive_sampling` function may alter the diagonals in-place to
            `torch.inf`, which would mess up the standard deviation calculation. Defaults to None.
        pos_idx (OptTensor, optional): index tensor where the value j at the ith position
            specifies the positive neighbor j for the ith input. This should be provided if the
            positive samples had to be determined, such as by using the `positive_sampling`
            function. For augmented genomes, the positive sample is already known.
            Defaults to None.
        scale (float, optional): negative exponential decay scale factor. Defaults to 7.0.
        no_negatives_mode (NO_NEGATIVES_MODES, optional): strategy for handling instances where
            no negatives exist, based on a semihard negative sampling strategy in which the
            negative is chosen to be closest to the positive AND farther from the anchor than
            the positive. If no negatives that fit the previous criteria exist,
            "closest_to_positive" will choose a sample that is closer to the anchor in the
            vicinity of the positive sample. Likewise, "closest_to_anchor" will choose a
            a negative sample that is the closest to the anchor in the embedding space. This
            issue should only arise for poorly trained models (such as at the beginning of
            training). Defaults to "closest_to_positive".

    Raises:
        ValueError: if pos_idx is not provided when output_embed_Y is None as this indicates
            positive samples were calculated prior to this
        ValueError: if an invalid strategy is passed for no_negatives_mode is passed. Valid
            options are "closest_to_positive" and "closest_to_anchor".

    Returns:
        PairTensor: (negative sample index tensor, negative sample weight tensor). The negative
            sample is a tensor of shape [N] where the value j at the ith position specifies the
            negative neighbor j for the ith input. The negative sample weight tensor is a
            tensor of shape [N] that is the triplet loss weight for each negative sample to
            downweight poor choices of negative genomes that are really close to the anchor.
    """
    if input_space_dist_std is None:
        input_space_dist_std = input_space_pairwise_dist.std()

    if output_embed_Y is None:
        # this chooses the neg sample from the real dataset
        # otherwise, Y is the augmented data, and this will choose from Y
        output_embed_Y = output_embed_X
        if pos_idx is None:
            raise ValueError("pos_idx must be supplied if a second tensor Y is not")
    else:
        # pos_idx is just aligned already in the case of sampling from point swapped
        # augmented data
        pos_idx = torch.arange(output_embed_Y.size(0)).to(output_embed_Y.device)

    batch_size = pos_idx.size(0)
    pos_idx = rearrange(pos_idx, "batch -> batch 1")

    # in case of sampling from real dataset, diag should be 0 due to self-comparison
    # so don't allow self to be chosen as neg sample
    # for pnt swap, generally diagonal should be the smallest, so chose another
    # negative sampling happens in the models embedding space,
    # not the input space
    embed_dist = torch.cdist(output_embed_X, output_embed_Y)

    # don't allow points closer than the pos sample be chosen
    # this also has the effect of removing self-comparisons
    # NOTE: checking distance to known positives in the OUTPUT embedding space
    # this allows dynamically choosing the negative samples as the model learns
    pos_dists = distance_from_index(embed_dist, pos_idx)

    gt_pos = embed_dist > pos_dists
    has_neg_choices = reduce(gt_pos, "batch1 batch2 -> batch1", "sum").bool()
    dist_from_pos = embed_dist - pos_dists

    # semi-hardish negative mining
    if has_neg_choices.all():
        # neg example is the one that is the next farther after the pos sample
        # dist = 0.0 is pos_idx, dist < 0.0 is closer than pos,
        # but we want just farther than pos
        neg_idx = _semi_hard_negative_sampling(
            dist_from_pos=dist_from_pos, pos_idx=pos_idx, batch_size=batch_size
        )
    else:
        neg_idx = pos_idx.new_full((batch_size,), fill_value=-1)
        has_neg_idx = has_neg_choices.nonzero(as_tuple=True)[0]
        no_neg_idx = has_neg_choices.logical_not().nonzero(as_tuple=True)[0]

        # calc negative for those with proper negative ie negative
        # is just the next set farther than the positive
        has_neg_dist_from_pos = dist_from_pos[has_neg_idx]
        has_neg_neg_idx = _semi_hard_negative_sampling(
            dist_from_pos=has_neg_dist_from_pos,
            pos_idx=pos_idx[has_neg_idx],
            batch_size=None,
        )
        neg_idx[has_neg_idx] = has_neg_neg_idx

        if no_negatives_mode == "closest_to_positive":
            no_neg_dist_from_pos = dist_from_pos[no_neg_idx].abs()
            no_neg_neg_idx = _semi_hard_negative_sampling(
                dist_from_pos=no_neg_dist_from_pos,
                pos_idx=pos_idx[no_neg_idx],
                batch_size=None,
            )
        elif no_negatives_mode == "closest_to_anchor":
            dist_from_anchor = embed_dist[no_neg_idx]
            # mask anchor
            dist_from_anchor[:, no_neg_idx] = torch.inf
            no_neg_neg_idx = dist_from_anchor.argmin(dim=-1)
        else:
            raise ValueError(
                f"Invalid strategy passed for no negatives: {no_negatives_mode=}"
            )
        neg_idx[no_neg_idx] = no_neg_neg_idx

    # calc negative sample weight
    neg_setwise_dists = input_space_pairwise_dist.gather(
        dim=-1, index=neg_idx.unsqueeze(1)
    ).squeeze()
    denom = 2 * (scale * input_space_dist_std) ** 2
    neg_weight = torch.exp(-neg_setwise_dists / denom)

    return neg_idx, neg_weight


def point_swap_sampling(
    batch: torch.Tensor,
    pos_idx: torch.Tensor,
    item_flow: torch.Tensor,
    sizes: torch.Tensor,
    sample_rate: float,
) -> PairTensor:
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
        sizes (torch.Tensor): shape: [batch_size], size of each point set
        sample_rate (float): rate of point swapping, (0.0, 1.0)

    Returns:
        PairTensor: augmented batch and the indices of the nodes that were
            swapped around
    """
    if sample_rate <= 0.0 or sample_rate >= 1.0:
        raise ValueError(f"Provided {sample_rate=} must be in the range (0.0, 1.0)")

    pos_idx = pos_idx.repeat_interleave(sizes).unsqueeze(1)
    aug_idx = item_flow.gather(dim=1, index=pos_idx).squeeze()

    # sample from uniform [0, 1) distribution to determine whether to swap
    # True = should swap, False = should not swap
    mask = torch.rand(aug_idx.size(0), device=aug_idx.device) < sample_rate
    default = torch.arange(aug_idx.size(0), device=aug_idx.device)
    aug_idx = torch.where(mask, aug_idx, default)

    augmented_batch = batch[aug_idx]
    return augmented_batch, aug_idx


def heuristic_augmented_negative_sampling(
    X_anchor: torch.Tensor,
    X_aug: torch.Tensor,
    y_aug: torch.Tensor,
    neg_idx: torch.Tensor,
    ptr: torch.Tensor,
    scale: float = 7.0,
) -> PairTensor:
    chamfer_distances: list[float] = list()
    for i, j in enumerate(neg_idx):
        j = int(j)

        start_i = ptr[i]
        stop_i = ptr[i + 1]
        start_j = ptr[j]
        stop_j = ptr[j + 1]

        set_i = X_anchor[start_i:stop_i]
        set_j = X_aug[start_j:stop_j]

        chamfer_dist = pairwise_chamfer_distance(set_i, set_j)
        chamfer_distances.append(chamfer_dist)

    aug_neg_dist = torch.tensor(chamfer_distances, device=X_anchor.device)
    ### AUG NEG WEIGHTS ###
    std = aug_neg_dist.std()
    denom = 2 * (scale * std) ** 2
    aug_neg_weights = torch.exp(-aug_neg_dist / denom)

    return y_aug[neg_idx], aug_neg_weights

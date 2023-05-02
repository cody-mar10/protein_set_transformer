from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from pst.utils.mask import compute_row_mask, pairwise_row_mask


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
    idx = torch.arange(embed_dist.size(0))

    # don't allow points closer than the pos sample be chosen
    # this also has the effect of removing self-comparisons
    pos_dists = embed_dist[idx, pos_idx].unsqueeze(1)

    gt_pos = embed_dist > pos_dists
    has_neg_choices = torch.sum(gt_pos, dim=-1).bool()

    # TODO: unclear if this is the best sampling option, but not I think having the negative examples be
    # weighted helps since there are no labels
    # sample from all indices where distance is greater than the positive example
    choice_idx, choice_opt = torch.where(gt_pos)
    choices = defaultdict(list)
    for i, j in zip(choice_idx, choice_opt):
        i = int(i)
        j = int(j)
        choices[i].append(j)

    if not has_neg_choices.all():
        # TODO: the above can cause a problem if the positive sample is somehow
        # farther away than the negative samples in the embedding space,
        # which may be common at the beginning of modeling
        # next best thing is to choose the 1st index smaller than the positive example
        # can just do:
        dist_from_pos = pos_dists - embed_dist
        dist_from_pos[torch.where(dist_from_pos <= 0.0)] = float("inf")
        # the closest negative will have the smallest nonzero distance
        idx_closest_to_pos = dist_from_pos.min(dim=-1)[1]
        for idx in range(embed_dist.size(0)):
            # only add the missing indices for now
            if idx not in choices:
                neg_ex = int(idx_closest_to_pos[idx])
                choices[idx].append(neg_ex)

    # need to keep this in order
    neg_idx = torch.tensor([random.choice(choices[idx]) for idx in range(len(choices))])
    # calc negative sample weight
    neg_setwise_dists = setwise_dist[idx, neg_idx]
    setwise_std = setwise_dist.std()
    denom = 2 * (scale * setwise_std) ** 2
    neg_weight = torch.exp(-neg_setwise_dists / denom)

    return neg_idx, neg_weight


class TripletSetSampler:
    """PointSet sampler for data augmentation during trainer. Samples positive examples
    from the original data vector space and samples negative examples from the
    embedding space of the model. This allows for online triplet sampling.
    """

    def __init__(self, setwise_distance: torch.Tensor) -> None:
        """Creates a data augmentation set sampler for a mini-batch of PointSets.

        Args:
            setwise_distance (torch.Tensor): symmetric set-wise distance matrix
        """
        self._setdist = setwise_distance

    def positive_sampling(self, mask_diagonal: bool = True) -> torch.Tensor:
        """Perform positive sampling using a set-wise distance metric like Chamfer distance.

        It is recommended that the set-wise distance be calculated BEFORE a model forward
        pass since this will be comparing data in the original dimension compared to the
        embedding dimension. The positive pairs in the embedding dimension will be heavily
        dependent on the model and how it is being trained and how it was initialized.

        Args:

            mask_diagonal (bool): use if the diagonal needs to be masked since that
                is likely a zero vector. In other words, don't choose self as the
                positive index. Defaults to True.

        Returns:
            torch.Tensor: tensor with the indices of the positive set
        """
        # choose smallest k distances
        # the diagonal is likely all 0s, so that will be the smallest values per row
        k = 2 if mask_diagonal else 1
        pos_sample_idx: torch.Tensor = self._setdist.topk(k, dim=-1, largest=False)[1][
            :, -1
        ]
        return pos_sample_idx

    def negative_sampling(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        *,
        pos_idx: Optional[torch.Tensor] = None,
        scale: float = 7.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return negative_sampling(self._setdist, X, Y, pos_idx=pos_idx, scale=scale)


class PointSwapSampler:
    _SENTINEL = -1

    def __init__(
        self,
        batch: torch.Tensor,
        pos_idx: torch.Tensor,
        item_flow: torch.Tensor,
        sample_rate: float,
        row_mask: Optional[torch.Tensor] = None,
    ):
        if sample_rate <= 0.0 or sample_rate > 1:
            raise ValueError(
                f"Provided {sample_rate=} -- must be in the range (0.0, 1.0]"
            )

        self._row_mask = compute_row_mask(batch) if row_mask is None else row_mask

        self._batch = batch
        self._item_flow = item_flow
        self._pos_idx = pos_idx
        self.sample_rate = sample_rate
        self._build_point_swap_index()

    def _build_point_swap_index(self):
        n_items_per_set = self._row_mask.sum(-1)
        # how many samples to draw per set based on how many real items there are
        n_samples_per_set = n_items_per_set.mul(self.sample_rate).ceil().long()

        samples: list[torch.Tensor] = list()
        for row, n_samples in zip(self._row_mask, n_samples_per_set):
            sample = torch.multinomial(row.float(), int(n_samples))
            samples.append(sample)

        # [b, n] where n is the maximum number of rows to swap
        # index (i, j) gets the index of item j to swap from set i
        # actual flow is stored in self._item_flow
        self.point_swap_idx = pad_sequence(
            samples, batch_first=True, padding_value=self._SENTINEL
        )

        # mask the point swapping idx to prevent swapping padded rows
        self.point_swap_mask = self.point_swap_idx != self._SENTINEL

    def point_swap(self) -> torch.Tensor:
        # clone batch so that the order of items in each set stays the same
        # other than for the items that get swapped
        # this will enable the use of positional encodings should the position
        # of the items in the set actually be important
        augmented_batch = self._batch.clone()

        # this holds the positive set index for all sets in the mini-batch
        for set_i, set_j in enumerate(self._pos_idx):
            mask = self.point_swap_mask[set_i]
            set_i_item_idx = self.point_swap_idx[set_i, mask]
            # item flow idx (i, j, m) = closest item in set j to item m in set i
            set_j_item_idx = self._item_flow[set_i, set_j, set_i_item_idx]
            # don't get from augmented batch since it will swap everything multiple times
            augmented_batch[set_i, set_i_item_idx] = self._batch[set_j, set_j_item_idx]

        return augmented_batch


def heuristic_augmented_negative_sampling(
    X_anchor: torch.Tensor,
    X_aug: torch.Tensor,
    y_aug: torch.Tensor,
    neg_idx: torch.Tensor,
    scale: float = 7.0,
    anchor_row_mask: Optional[torch.Tensor] = None,
    average: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_aug_neg = X_aug[neg_idx]

    ### CALC SETWISE DIST ###
    # X: [b, n, d] -> neg_dist: [b, n, n]
    neg_pairwise_dist = torch.cdist(X_anchor, X_aug_neg, p=2)
    if anchor_row_mask is None:
        anchor_row_mask = compute_row_mask(X_anchor)

    augmented_row_mask = compute_row_mask(X_aug_neg)

    full_mask = pairwise_row_mask(anchor_row_mask, augmented_row_mask)
    filled_mask = torch.where(full_mask, 0.0, torch.inf)
    neg_pairwise_dist += filled_mask

    aug_neg_chamfer_dist = torch.zeros(X_anchor.size(0))
    for mask, dim in zip([anchor_row_mask, augmented_row_mask], [-1, -2]):
        # find closest item
        min_dist: torch.Tensor = torch.min(neg_pairwise_dist, dim=dim)[0]
        # don't let padded rows/items count
        min_dist[torch.where(min_dist == torch.inf)] = 0.0
        item_dist = min_dist.sum(dim=-1)
        if average:
            item_dist /= mask.sum(dim=-1)

        aug_neg_chamfer_dist += item_dist

    ### AUG NEG WEIGHTS ###
    # TODO: using std here may make the weights really small since there are fewer distances
    # gonna use mean for now
    denom = 2 * (scale * aug_neg_chamfer_dist.mean())
    aug_neg_weights = torch.exp(-aug_neg_chamfer_dist / denom)

    return y_aug[neg_idx], aug_neg_weights

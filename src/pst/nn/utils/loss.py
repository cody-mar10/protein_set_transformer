from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from pst.data.utils import graph_sizes_to_index_pointer
from pst.typing import OptTensor


def average(x: torch.Tensor) -> torch.Tensor:
    return einops.reduce(x, "batch -> ", "mean")


class WeightedTripletLoss(nn.Module):
    def __init__(self, margin: float) -> None:
        super(WeightedTripletLoss, self).__init__()
        self.margin = margin
        self.loss_minimum = torch.tensor(0.0)

    @staticmethod
    def squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        squared_diff = torch.pow(x - y, 2)
        return einops.reduce(squared_diff, "batch dim -> batch", "sum")

    def forward(
        self,
        y_self: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        weights: torch.Tensor,
        reduce: bool = False,
        class_weights: OptTensor = None,
    ) -> torch.Tensor:
        if class_weights is None:
            # no actual weights
            class_weights = y_self.new_ones(y_self.size(0))

        positive_dist = self.squared_distance(y_self, y_pos)
        negative_dist = self.squared_distance(y_self, y_neg).mul(weights)

        dist = positive_dist - negative_dist + self.margin

        # minimum dist is 0 for the perfect case
        # class weights also rescale contribution to loss for a weighted average
        # weights are 1.0 for most common class, and all rarer classes are > 1.0
        # this has the effect of amplifying bad performance for rare classes
        triplet_loss = dist.maximum(self.loss_minimum) * class_weights

        if reduce:
            return average(triplet_loss)
        return triplet_loss


class AugmentedWeightedTripletLoss(nn.Module):
    def __init__(self, margin: float) -> None:
        super(AugmentedWeightedTripletLoss, self).__init__()
        self.loss_fn = WeightedTripletLoss(margin)

    def forward(
        self,
        y_self: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        neg_weights: torch.Tensor,
        class_weights: OptTensor = None,
        y_aug_pos: OptTensor = None,
        y_aug_neg: OptTensor = None,
        aug_neg_weights: OptTensor = None,
    ) -> torch.Tensor:
        loss: torch.Tensor = self.loss_fn(
            y_self=y_self,
            y_pos=y_pos,
            y_neg=y_neg,
            weights=neg_weights,
            reduce=False,
            class_weights=class_weights,
        )

        if y_aug_pos is not None:
            if y_aug_neg is None or aug_neg_weights is None:
                raise ValueError("Should also provide y_aug_neg and aug_neg_weights")
            augmented_loss: torch.Tensor = self.loss_fn(
                y_self=y_self,
                y_pos=y_aug_pos,
                y_neg=y_aug_neg,
                weights=aug_neg_weights,
                reduce=False,
                class_weights=class_weights,
            )
            loss += augmented_loss
            return average(loss) / 2
        return average(loss)


class MaskedLanguageModelingLoss(nn.Module):
    def __init__(self, masking_rate: float):
        super().__init__()
        if masking_rate < 0 or masking_rate > 0.5:
            raise ValueError("masking_rate should be in [0.0, 0.5]")

        # this is PER scaffold
        self.masking_rate = masking_rate

    def mask(self, sizes: torch.Tensor, ptr: OptTensor = None) -> torch.Tensor:
        # sizes shape: [num scaffolds]
        # ptr shape:   [num scaffolds + 1]
        # mask shape:  [num proteins]

        if ptr is None:
            ptr = graph_sizes_to_index_pointer(sizes)

        mask = torch.zeros(int(sizes.sum()), device=sizes.device, dtype=torch.bool)
        for idx, size in enumerate(sizes):
            view = mask[ptr[idx] : ptr[idx + 1]]
            # guarantee that at least one element is masked
            while True:
                scaffold_mask: torch.Tensor = torch.rand(size, device=sizes.device) < self.masking_rate  # type: ignore
                if scaffold_mask.any():
                    break

            view[:] = scaffold_mask

        return mask

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # compute euclidean distance (sort of) between masked (expected) embeddings
        # and the model's predicted embeddings
        return F.mse_loss(y_pred, y_target, reduction="mean")

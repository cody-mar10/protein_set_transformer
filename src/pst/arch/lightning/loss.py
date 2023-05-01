from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class WeightedTripletLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super(WeightedTripletLoss, self).__init__()
        self.alpha = alpha
        self.loss_minimum = torch.tensor(0.0)

    def forward(
        self,
        y_self: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        weights: torch.Tensor,
        reduce: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if class_weights is None:
            # no actual weights
            class_weights = torch.ones(y_self.size(0))

        positive_dist = torch.pow(y_self - y_pos, 2).sum(dim=-1)
        negative_dist = torch.pow(y_self - y_neg, 2).sum(dim=-1).mul(weights)

        # minimum dist is 0 for the perfect case
        # class weights also rescale contribution to loss for a weighted average
        # weights are 1.0 for most common class, and all rarer classes are > 1.0
        # this has the effect of amplifying bad performance for rare classes
        dist = (
            torch.minimum(positive_dist - negative_dist + self.alpha, self.loss_minimum)
            * class_weights
        )
        if reduce:
            return torch.mean(dist, dim=0)
        return dist


class AugmentedWeightedTripletLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super(AugmentedWeightedTripletLoss, self).__init__()
        self.alpha = alpha
        self.loss_fn = WeightedTripletLoss(alpha)

    def forward(
        self,
        y_self: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        neg_weights: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        y_aug_pos: Optional[torch.Tensor] = None,
        y_aug_neg: Optional[torch.Tensor] = None,
        aug_neg_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss: torch.Tensor = self.loss_fn(
            y_self, y_pos, y_neg, neg_weights, class_weights
        )

        if y_aug_pos is not None:
            if y_aug_neg is None or aug_neg_weights is None:
                raise ValueError("Should also provide y_aug_neg and aug_neg_weights")
            augmented_loss: torch.Tensor = self.loss_fn(
                y_self,
                y_aug_pos,
                y_aug_neg,
                aug_neg_weights,
                class_weights,
            )
            loss += augmented_loss
        return loss.mean(dim=0) / 2

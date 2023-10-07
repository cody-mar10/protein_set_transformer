from __future__ import annotations

from typing import Optional

import einops
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class LossConfig(BaseModel):
    margin: float = Field(0.1, description="triplet loss margin", gt=0.0)


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
        class_weights: Optional[torch.Tensor] = None,
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
        class_weights: Optional[torch.Tensor] = None,
        y_aug_pos: Optional[torch.Tensor] = None,
        y_aug_neg: Optional[torch.Tensor] = None,
        aug_neg_weights: Optional[torch.Tensor] = None,
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

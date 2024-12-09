import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def forward(
        self,
        y_pred: torch.Tensor,
        masked_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        y_pos: OptTensor = None,
        y_pos_weight: OptTensor = None,
    ) -> torch.Tensor:
        y_pred = y_pred[node_mask]

        # compute euclidean distance (sort of) between masked (expected) embeddings
        # and the model's predicted embeddings
        loss = F.mse_loss(y_pred, masked_embeddings, reduction="mean")

        if y_pos is not None:
            if y_pos_weight is None:
                raise ValueError(
                    "Should also provide y_pos_weight when provided positive samples"
                )

            y_pos = y_pos[node_mask]
            y_pos_weight = y_pos_weight[node_mask]

            # weighted mse where the choice of positive is weighted to penalize bad choices
            # of positives
            pos_loss: torch.Tensor = (
                torch.abs(y_pos - masked_embeddings)
                .square()
                .mul(y_pos_weight.unsqueeze(-1))
                .mean()
            )

            loss += pos_loss

        return loss

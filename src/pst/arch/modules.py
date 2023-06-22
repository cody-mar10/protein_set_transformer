from __future__ import annotations

from typing import Any, cast, Literal

import lightning as L
import torch
from transformers import get_linear_schedule_with_warmup

from pst._typing import DataBatch, OptionalAttentionOutput
from pst.model import SetTransformer
from .training.distance import stacked_batch_chamfer_distance
from .training.loss import AugmentedWeightedTripletLoss
from .training.sampling import (
    heuristic_augmented_negative_sampling,
    negative_sampling,
    point_swap_sampling,
    positive_sampling,
)

_STAGE_TYPE = Literal["train", "val", "test"]


class ProteinSetTransformer(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        n_enc_layers: int,
        multiplier: float,
        dropout: float,
        optimizer_kwargs: dict[str, Any],
        loss_kwargs: dict[str, Any],
        augmentation_kwargs: dict[str, Any],
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SetTransformer(
            in_dim=in_dim,
            out_dim=out_dim,
            heads=num_heads,
            n_enc_layers=n_enc_layers,
            multiplier=multiplier,
            dropout=dropout,
        )

        if compile:
            self.model = torch.compile(self.model)

        self.criterion = AugmentedWeightedTripletLoss(**loss_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.augmentation_kwargs = augmentation_kwargs

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.optimizer_kwargs["lr"],
            betas=self.optimizer_kwargs["betas"],
            weight_decay=self.optimizer_kwargs["weight_decay"],
            eps=1e-7 if self.trainer.precision == "16-mixed" else 1e-8,
        )
        config: dict[str, Any] = {"optimizer": optimizer}
        if self.optimizer_kwargs.get("use_scheduler", True):
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.optimizer_kwargs["warmup_steps"],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return config

    def _log_loss(self, loss: torch.Tensor, stage: _STAGE_TYPE):
        self.log(
            f"{stage}_loss",
            value=loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> OptionalAttentionOutput:
        output = self.model(
            x=x,
            edge_index=edge_index,
            ptr=ptr,
            return_attention_weights=return_attention_weights,
        )
        return output

    def _databatch_forward(
        self, batch: DataBatch, return_attention_weights: bool = False
    ) -> OptionalAttentionOutput:
        return self(
            x=batch.x,
            edge_index=batch.edge_index,
            ptr=batch.ptr,
            return_attention_weights=return_attention_weights,
        )

    def _forward_step(
        self,
        batch: DataBatch,
        batch_idx: int,
        stage: _STAGE_TYPE,
        augment_data: bool = True,
    ) -> torch.Tensor:
        setwise_dist, item_flow = stacked_batch_chamfer_distance(
            batch=batch.x, ptr=batch.ptr
        )

        #### REAL DATA ####
        # positive mining
        pos_idx = positive_sampling(setwise_dist)

        # forward pass
        y_anchor = cast(
            torch.Tensor,
            self._databatch_forward(batch=batch, return_attention_weights=False),
        )

        # negative sampling
        scale = self.augmentation_kwargs["sample_scale"]
        neg_idx, neg_weights = negative_sampling(
            setwise_dist=setwise_dist,
            X=y_anchor,
            pos_idx=pos_idx,
            scale=scale,
        )

        y_pos = y_anchor[pos_idx]
        y_neg = y_anchor[neg_idx]

        if augment_data:
            y_aug_pos, y_aug_neg, aug_neg_weights = self._augmented_forward_step(
                batch=batch,
                pos_idx=pos_idx,
                neg_idx=neg_idx,
                item_flow=item_flow,
            )
        else:
            y_aug_pos = None
            y_aug_neg = None
            aug_neg_weights = None

        loss: torch.Tensor = self.criterion(
            y_self=y_anchor,
            y_pos=y_pos,
            y_neg=y_neg,
            neg_weights=neg_weights,
            class_weights=batch.weight,
            y_aug_pos=y_aug_pos,
            y_aug_neg=y_aug_neg,
            aug_neg_weights=aug_neg_weights,
        )

        self._log_loss(loss=loss, stage=stage)
        return loss

    def _augmented_forward_step(
        self,
        batch: DataBatch,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
        item_flow: dict[tuple[int, int], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_batch = point_swap_sampling(
            batch=batch.x,
            ptr=batch.ptr,
            pos_idx=pos_idx,
            item_flow=item_flow,
            sample_rate=self.augmentation_kwargs["sample_rate"],
        )

        y_aug_pos = self(
            x=augmented_batch,
            edge_index=batch.edge_index,
            ptr=batch.ptr,
            return_attention_weights=False,
        )

        y_aug_neg, aug_neg_weights = heuristic_augmented_negative_sampling(
            X_anchor=batch.x,
            X_aug=augmented_batch,
            y_aug=y_aug_pos,
            neg_idx=neg_idx,
            ptr=batch.ptr,
            scale=self.augmentation_kwargs["sample_scale"],
        )

        return y_aug_pos, y_aug_neg, aug_neg_weights

    def training_step(self, train_batch: DataBatch, batch_idx: int) -> torch.Tensor:
        return self._forward_step(
            batch=train_batch,
            batch_idx=batch_idx,
            stage="train",
            augment_data=True,
        )

    def validation_step(self, val_batch: DataBatch, batch_idx: int) -> torch.Tensor:
        return self._forward_step(
            batch=val_batch,
            batch_idx=batch_idx,
            stage="val",
            augment_data=False,
        )

    def test_step(self, test_batch: DataBatch, batch_idx: int) -> torch.Tensor:
        return self._forward_step(
            batch=test_batch,
            batch_idx=batch_idx,
            stage="test",
            augment_data=False,
        )

    def predict_step(
        self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        return self._databatch_forward(batch=batch)

from __future__ import annotations

from typing import Any, Iterator, Literal, cast

import lightning as L
import torch
from lightning_cv import BaseModelConfig, CrossValModuleMixin
from pydantic import BaseModel, Field
from torch.nn.parameter import Parameter
from transformers import get_linear_schedule_with_warmup

from pst._typing import DataBatch, OptionalAttentionOutput
from pst.model import SetTransformer

from .training.distance import stacked_batch_chamfer_distance
from .training.loss import AugmentedWeightedTripletLoss, LossConfig
from .training.sampling import (
    AugmentationConfig,
    heuristic_augmented_negative_sampling,
    negative_sampling,
    point_swap_sampling,
    positive_sampling,
)

_STAGE_TYPE = Literal["train", "val", "test"]


class OptimizerConfig(BaseModel):
    lr: float = Field(1e-3, description="learning rate", ge=1e-5, le=1e-1)
    weight_decay: float = Field(
        0.0, description="optimizer weight decay", ge=0.0, le=1e-1
    )
    betas: tuple[float, float] = Field((0.9, 0.999), description="optimizer betas")
    warmup_steps: int = Field(0, description="number of warmup steps", ge=0)
    use_scheduler: bool = Field(
        False, description="whether or not to use a linearly decaying scheduler"
    )


class ModelConfig(BaseModelConfig):
    in_dim: int = Field(
        -1, description="input dimension, default is to use the dataset dimension"
    )
    out_dim: int = Field(
        -1, description="output dimension, default is to use the input dimension"
    )
    num_heads: int = Field(4, description="number of attention heads", gt=0)
    n_enc_layers: int = Field(5, description="number of encoder layers", gt=0)
    multiplier: float = Field(
        1.0,
        description=(
            "multiplicative weight to de-emphasize (< 1.0) or over-emphasize "
            "(> 1.0) protein weights when decoding a genome representation by "
            "pooling over all proteins in a genome"
        ),
        gt=0.0,
    )
    dropout: float = Field(
        0.5, description="dropout proportion during training", ge=0.0, lt=1.0
    )
    compile: bool = Field(False, description="compile model using torch.compile")
    optimizer: OptimizerConfig
    loss: LossConfig
    augmentation: AugmentationConfig

    @classmethod
    def default(cls):
        schema = dict()
        for key, field in cls.model_fields.items():
            if isinstance(field.annotation, type):
                if isinstance(field.annotation, BaseModel) or issubclass(
                    field.annotation, BaseModel
                ):
                    # field is a pydantic model
                    # NOTE: does not handle any nesting below this...
                    value = field.annotation.model_construct()
                else:
                    value = field.get_default()

                schema[key] = value

        return cls.model_validate(schema)


class ProteinSetTransformer(L.LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump(exclude={"fabric"}))
        self.model = SetTransformer(
            **self.config.model_dump(
                include={
                    "in_dim",
                    "out_dim",
                    "num_heads",
                    "n_enc_layers",
                    "multiplier",
                    "dropout",
                }
            )
        )

        if self.config.compile:
            self.model = torch.compile(self.model)

        self.criterion = AugmentedWeightedTripletLoss(**config.loss.model_dump())
        self.optimizer_cfg = config.optimizer
        self.augmentation_cfg = config.augmentation
        self.fabric = config.fabric

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.model.parameters(recurse=recurse)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.optimizer_cfg.lr,
            betas=self.optimizer_cfg.betas,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        config: dict[str, Any] = {"optimizer": optimizer}
        if self.optimizer_cfg.use_scheduler:
            if self.fabric is None:
                self.estimated_steps = self.trainer.estimated_stepping_batches

            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.optimizer_cfg.warmup_steps,
                num_training_steps=self.estimated_steps,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        return config

    def _log_loss(self, loss: torch.Tensor, batch_size: int, stage: _STAGE_TYPE):
        self.log(
            f"{stage}_loss",
            value=loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
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
        batch_size = batch.setsize.numel()
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
        scale = self.augmentation_cfg.sample_scale
        neg_idx, neg_weights = negative_sampling(
            setwise_dist=setwise_dist,
            X=y_anchor,
            pos_idx=pos_idx,
            scale=scale,
            no_negatives_mode=self.augmentation_cfg.no_negatives_mode,
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

        if self.fabric is None:
            self._log_loss(loss=loss, batch_size=batch_size, stage=stage)
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
            sample_rate=self.augmentation_cfg.sample_rate,
        )

        y_aug_pos = self(
            x=augmented_batch,
            edge_index=batch.edge_index,
            ptr=batch.ptr,
            return_attention_weights=False,
        )
        # TODO: fix error with min reduction
        y_aug_neg, aug_neg_weights = heuristic_augmented_negative_sampling(
            X_anchor=batch.x,
            X_aug=augmented_batch,
            y_aug=y_aug_pos,
            neg_idx=neg_idx,
            ptr=batch.ptr,
            scale=self.augmentation_cfg.sample_scale,
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


class CrossValPST(CrossValModuleMixin, ProteinSetTransformer):
    __error_msg__ = (
        "Model {stage} is not allowed during cross validation. Only training and "
        "validation is supported."
    )

    def __init__(self, config: ModelConfig):
        ProteinSetTransformer.__init__(self, config=config)

        # needed for type hints
        self.fabric: L.Fabric
        CrossValModuleMixin.__init__(self, config=config)

    def test_step(self, test_batch: DataBatch, batch_idx: int):
        raise RuntimeError(self.__error_msg__.format(stage="testing"))

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0):
        raise RuntimeError(self.__error_msg__.format(stage="inference"))

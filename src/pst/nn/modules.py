from __future__ import annotations

from typing import Any, Iterator, Literal

import lightning as L
import torch
from lightning_cv import BaseModelConfig, CrossValModuleMixin
from pydantic import BaseModel, Field
from torch.nn.parameter import Parameter
from transformers import get_linear_schedule_with_warmup

from pst.data.modules import GenomeDataset
from pst.typing import GenomeGraphBatch, OptGraphAttnOutput, OptTensor

from .layers import PositionalEmbedding, PositionwiseFeedForward
from .models import SetTransformer
from .utils.distance import stacked_batch_chamfer_distance
from .utils.loss import AugmentedWeightedTripletLoss, LossConfig
from .utils.sampling import (
    AugmentationConfig,
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
    embed_scale: int = Field(
        4,
        description=(
            "scale factor for positional and strand embeddings. These embeddings will be "
            "of size in_dim/n, ie higher number means smaller embeddings."
        ),
        ge=1,
        le=8,
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
        if config.out_dim == -1:
            config.out_dim = config.in_dim

        self.config = config
        self.save_hyperparameters(config.model_dump(exclude={"fabric"}))

        # 2048 ptns should be large enough for probably all viruses
        embedding_dim = config.in_dim // config.embed_scale
        self.positional_embedding = PositionalEmbedding(
            dim=embedding_dim, max_size=2048
        )

        # embed +/- gene strand
        self.strand_embedding = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=embedding_dim
        )

        # plm embeddings, positional embeddings, and strand embeddings
        # will be concatenated together and then projected back to the original dim
        self.proj = PositionwiseFeedForward(
            in_dim=config.in_dim + 2 * embedding_dim,
            out_dim=config.in_dim,
            dropout=config.dropout,
        )

        self.model = SetTransformer(
            **self.config.model_dump(
                include={
                    "in_dim",
                    "out_dim",
                    "num_heads",
                    "n_enc_layers",
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

    def check_max_size(self, dataset: GenomeDataset):
        if dataset.max_size > self.positional_embedding.max_size:
            self.positional_embedding.expand(dataset.max_size)

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
        batch: OptTensor = None,
        return_attention_weights: bool = False,
    ) -> OptGraphAttnOutput:
        output = self.model(
            x=x,
            edge_index=edge_index,
            ptr=ptr,
            batch=batch,
            return_attention_weights=return_attention_weights,
        )
        return output

    def _databatch_forward(
        self,
        batch: GenomeGraphBatch,
        return_attention_weights: bool = False,
        x: OptTensor = None,
    ) -> OptGraphAttnOutput:
        if x is None:
            x = batch.x

        return self(
            x=x,
            edge_index=batch.edge_index,
            ptr=batch.ptr,
            batch=batch.batch,
            return_attention_weights=return_attention_weights,
        )

    def _forward_step(
        self,
        batch: GenomeGraphBatch,
        batch_idx: int,
        stage: _STAGE_TYPE,
        augment_data: bool = True,
    ) -> torch.Tensor:
        # TODO: strand and pos embeddings are dominating signal
        # this is especially bad here since the chamfer distance and triplet sampling will
        # be heavily influenced by the strand embeddings
        # I want the plm embeddings to be the dominant signal

        # add strand encoding here so augmented data has strand info already
        strand_embed = self.strand_embedding(batch.strand)
        positional_embed = self.positional_embedding(batch.pos.squeeze())

        # calculate chamfer distance only based on the plm embeddings
        # want to maximize that signal over strand and positional embeddings
        batch_size = batch.num_proteins.numel()
        setwise_dist, item_flow = stacked_batch_chamfer_distance(
            batch=batch.x, ptr=batch.ptr
        )

        #### REAL DATA ####
        # positive mining
        pos_idx = positive_sampling(setwise_dist)

        # adding positional and strand embeddings lead to those dominating the plm signal
        # we can concatenate them here then use a linear layer to project down back to
        # the original feature dim and force the model to directly learn which of these
        # are most important

        # NOTE: we do not adjust the original data at batch.x
        # this lets the augmented data adjust the positional and strand embeddings
        # independently of the original data
        x = self._incorporate_positional_and_strand_embeddings(
            x=batch.x,
            positional_embed=positional_embed,
            strand_embed=strand_embed,
        )

        # forward pass
        y_anchor, _ = self._databatch_forward(
            batch=batch,
            return_attention_weights=False,
            x=x,
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
                y_anchor=y_anchor,
                item_flow=item_flow,
                positional_embed=positional_embed,
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
        batch: GenomeGraphBatch,
        pos_idx: torch.Tensor,
        y_anchor: torch.Tensor,
        item_flow: torch.Tensor,
        positional_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_batch, aug_idx = point_swap_sampling(
            batch=batch.x,
            pos_idx=pos_idx,
            item_flow=item_flow,
            sizes=batch.num_proteins,
            sample_rate=self.augmentation_cfg.sample_rate,
        )

        # let strand use original indices strands
        strand = batch.strand[aug_idx]
        strand_embed = self.strand_embedding(strand)

        # however instead of changing the positional idx, just keep the same
        # this is basically attempting to mirror same protein encoded in a diff position
        x_aug = self._incorporate_positional_and_strand_embeddings(
            x=augmented_batch,
            positional_embed=positional_embed,
            strand_embed=strand_embed,
        )

        y_aug_pos, _ = self._databatch_forward(
            batch=batch,
            return_attention_weights=False,
            x=x_aug,
        )

        # TODO: hparam to choose true negative sampling or heuristic
        # NOTE: computing chamfer distance without positional or strand info
        setdist_real_aug, _ = stacked_batch_chamfer_distance(
            batch=batch.x, ptr=batch.ptr, other=augmented_batch
        )

        aug_neg_idx, aug_neg_weights = negative_sampling(
            setwise_dist=setdist_real_aug,
            X=y_anchor,
            Y=y_aug_pos,
            scale=self.augmentation_cfg.sample_scale,
            no_negatives_mode=self.augmentation_cfg.no_negatives_mode,
        )

        y_aug_neg = y_aug_pos[aug_neg_idx]
        return y_aug_pos, y_aug_neg, aug_neg_weights

    def _incorporate_positional_and_strand_embeddings(
        self,
        x: torch.Tensor,
        positional_embed: torch.Tensor,
        strand_embed: torch.Tensor,
    ) -> torch.Tensor:
        x_cat = torch.cat((x, positional_embed, strand_embed), dim=-1)
        return self.proj(x_cat)

    def training_step(
        self, train_batch: GenomeGraphBatch, batch_idx: int
    ) -> torch.Tensor:
        return self._forward_step(
            batch=train_batch,
            batch_idx=batch_idx,
            stage="train",
            augment_data=True,
        )

    def validation_step(
        self, val_batch: GenomeGraphBatch, batch_idx: int
    ) -> torch.Tensor:
        return self._forward_step(
            batch=val_batch,
            batch_idx=batch_idx,
            stage="val",
            augment_data=False,
        )

    def test_step(self, test_batch: GenomeGraphBatch, batch_idx: int) -> torch.Tensor:
        return self._forward_step(
            batch=test_batch,
            batch_idx=batch_idx,
            stage="test",
            augment_data=False,
        )

    def predict_step(
        self, batch: GenomeGraphBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> OptGraphAttnOutput:
        strand_embed = self.strand_embedding(batch.strand)
        positional_embed = self.positional_embedding(batch.pos.squeeze())

        x_with_pos_and_strand = self._incorporate_positional_and_strand_embeddings(
            x=batch.x,
            positional_embed=positional_embed,
            strand_embed=strand_embed,
        )

        # internally, the positional embeddings are added later
        return self._databatch_forward(
            batch=batch, x=x_with_pos_and_strand, return_attention_weights=True
        )


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

    def test_step(self, test_batch: GenomeGraphBatch, batch_idx: int):
        raise RuntimeError(self.__error_msg__.format(stage="testing"))

    def predict_step(
        self, batch: GenomeGraphBatch, batch_idx: int, dataloader_idx: int = 0
    ):
        raise RuntimeError(self.__error_msg__.format(stage="inference"))

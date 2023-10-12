from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import lightning as L
import torch
from lightning_cv import CrossValModuleMixin
from transformers import get_linear_schedule_with_warmup

from pst.data.modules import GenomeDataset
from pst.nn.config import ModelConfig
from pst.nn.layers import PositionalEmbedding
from pst.nn.models import SetTransformer, SetTransformerDecoder, SetTransformerEncoder
from pst.nn.utils.distance import stacked_batch_chamfer_distance
from pst.nn.utils.loss import AugmentedWeightedTripletLoss
from pst.nn.utils.sampling import (
    negative_sampling,
    point_swap_sampling,
    positive_sampling,
)
from pst.typing import GenomeGraphBatch, OptGraphAttnOutput, OptTensor

_STAGE_TYPE = Literal["train", "val", "test"]


class ProteinSetTransformer(L.LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.out_dim == -1:
            config.out_dim = config.in_dim

        self.config = deepcopy(config)
        self.save_hyperparameters(config.model_dump(exclude={"fabric"}))

        embedding_dim = config.in_dim // config.embed_scale

        # 2048 ptns should be large enough for probably all viruses
        self.positional_embedding = PositionalEmbedding(
            dim=embedding_dim, max_size=2048
        )

        # embed +/- gene strand
        self.strand_embedding = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=embedding_dim
        )

        self.config.in_dim += 2 * embedding_dim

        if not config.proj_cat:
            # plm embeddings, positional embeddings, and strand embeddings
            # will be concatenated together and then projected back to the original dim
            # by the first attention layer, BUT if we don't want that
            # then the output dimension will be equal to the original feature dim
            # plus the dim for both the positional and strand embeddings
            self.config.out_dim = self.config.in_dim

        self.model = SetTransformer(
            **self.config.model_dump(
                include={
                    "in_dim",
                    "out_dim",
                    "num_heads",
                    "n_enc_layers",
                    "dropout",
                    "layer_dropout",
                }
            )
        )

        if self.config.compile:
            self.model: SetTransformer = torch.compile(self.model)  # type: ignore

        self.criterion = AugmentedWeightedTripletLoss(**config.loss.model_dump())
        self.optimizer_cfg = config.optimizer
        self.augmentation_cfg = config.augmentation
        self.fabric = config.fabric

    def check_max_size(self, dataset: GenomeDataset):
        if dataset.max_size > self.positional_embedding.max_size:
            self.positional_embedding.expand(dataset.max_size)

    @property
    def encoder(self) -> SetTransformerEncoder:
        return self.model.encoder

    @property
    def decoder(self) -> SetTransformerDecoder:
        return self.model.decoder

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
        # we can concatenate them here, then use a linear layer to project down back to
        # the original feature dim and force the model to directly learn which of these
        # are most important

        # NOTE: we do not adjust the original data at batch.x
        # this lets the augmented data adjust the positional and strand embeddings
        # independently of the original data
        x = self._concatenate_embeddings(
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

        # let strand use original strand for each ptn
        strand = batch.strand[aug_idx]
        strand_embed = self.strand_embedding(strand)

        # however instead of changing the positional idx, just keep the same
        # this is basically attempting to mirror same protein encoded in a diff position
        x_aug = self._concatenate_embeddings(
            x=augmented_batch,
            positional_embed=positional_embed,
            strand_embed=strand_embed,
        )

        y_aug_pos, _ = self._databatch_forward(
            batch=batch,
            return_attention_weights=False,
            x=x_aug,
        )

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

    def _concatenate_embeddings(
        self,
        x: torch.Tensor,
        positional_embed: torch.Tensor,
        strand_embed: torch.Tensor,
    ) -> torch.Tensor:
        x_cat = torch.cat((x, positional_embed, strand_embed), dim=-1)
        return x_cat

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

        x_with_pos_and_strand = self._concatenate_embeddings(
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

from __future__ import annotations

from typing import Any, Literal, Optional

import lightning as L
import torch
from torch import optim
from transformers import get_linear_schedule_with_warmup

from .distance import batch_chamfer_distance
from .loss import AugmentedWeightedTripletLoss
from .sampling import negative_sampling, PointSwapSampler, TripletSetSampler
from pst.arch.model import SetTransformer, AttentionSchema
from pst.utils.mask import compute_row_mask, row_mask_to_attn_mask
from pst.utils._types import BatchType


class _ProteinSetTransformer(L.LightningModule):
    def __init__(
        self,
        # model
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_heads: int,
        n_outputs: int = 1,
        num_indices: int = 32,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        norm: bool = True,
        *,
        # optimizer
        warmup_steps: int = 5000,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        use_scheduler: bool = True,
        # point swap sampling
        sample_scale: float = 7.0,
        sample_rate: float = 0.5,
        # loss
        loss_margin: float = 0.1,
        compile: bool = False,
    ) -> None:
        """ProteinSetTransformer LightningModule. See SetTransformer for implementation details.

        Args:
            Model:
                in_dim (int): input data feature dimension
                out_dim (int): output data embedding dimension
                hidden_dim (int): hidden layer embedding dimension
                num_heads (int): number of attention heads
                num_indices (int, optional): projection dimension for large set efficient multiheaded attention. Defaults to 32.
                n_enc_layers (int, optional): number of encoder layers. Defaults to 2.
                n_dec_layers (int, optional): number of decoder layers, not including a pooling attention layer at the beginning and the fully connected layers at the end. Defaults to 2.
                dropout (float, optional): dropout probability during training. Defaults to 0.0.
                bias (bool, optional): Include bias in linear layers. Defaults to True.
                norm (bool, optional): Include a LayerNorm operation after each attention block. Defaults to True.

            Optimizer:
                warmup_steps (int, optional): number of steps to warm up learning rate
                    to max learning rate. Defaults to 5000.
                lr (float, optional): optimizer learning rate. Defaults to 1e-3.
                betas (tuple[float, float], optional): optimizer beta values. Defaults to (0.9, 0.999).
                weight_decay (float, optional): optimizer weight decay. Defaults to 0.0.

            PointSwap Sampler:
                sample_scale (float, optional): PointSwap sampler distance exponential decay scale factor. Smaller numbers force the model to increase the distance between positive and negative pairs, and vice versa. Defaults to 7.0.
                sample_rate (float, optional): PointSwap sampler swapping rate. Defaults to 0.5, ie for a positive pair, swap 50% of proteins.

            Loss:
                loss_alpha (float, optional): Constant additive term in loss calculation. Defaults to 0.1.
        """
        super().__init__()
        self.save_hyperparameters(ignore="precomputed_sampling")

        self.criterion = AugmentedWeightedTripletLoss(loss_margin)
        self.model = SetTransformer(
            in_dim=in_dim,
            n_outputs=n_outputs,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_indices=num_indices,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            dropout=dropout,
            bias=bias,
            normalize_Q=norm,
        )
        if compile:
            self.model = torch.compile(self.model)

    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        # return self.model(X, **kwargs)
        raise NotImplementedError(
            "You instantiate a concrete subclass like GenomeClassifier or GeneClassifier"
        )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            betas=self.hparams["betas"],
            weight_decay=self.hparams["weight_decay"],
            eps=1e-7 if self.trainer.precision == "16-mixed" else 1e-8,
        )
        config: dict[str, Any] = {"optimizer": optimizer}
        if self.hparams["use_scheduler"]:
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams["warmup_steps"],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        return config

    def _log_loss(self, loss: torch.Tensor, stage: Literal["train", "val", "test"]):
        self.log(
            f"{stage}_loss",
            value=loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        forward_kwargs = dict(return_weights=False, attn_mask=None)
        return self(batch, **forward_kwargs)

    def _augmented_forward_step(
        self,
        batch: torch.Tensor,
        y_anchor: torch.Tensor,
        pos_idx: torch.Tensor,
        item_flow: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_batch = PointSwapSampler(
            batch=batch,
            pos_idx=pos_idx,
            item_flow=item_flow,
            sample_rate=self.hparams["sample_rate"],
            row_mask=row_mask,
        ).point_swap()

        y_aug_pos = self._forward(augmented_batch)
        # TODO: i think row_mask should be the same for regular batch and augmented batch
        aug_row_mask = compute_row_mask(augmented_batch)
        # TODO: the distances need to be between each anchor set and each augmented set, so not in between...
        # pros: don't need to consider item flow at all since I just need the distances
        # cons: think this may be harder to implement?
        aug_setwise_dist, _ = batch_chamfer_distance(
            X=augmented_batch, row_mask=aug_row_mask
        )
        scale = self.hparams["sample_scale"]
        aug_neg_idx, aug_neg_weights = negative_sampling(
            setwise_dist=aug_setwise_dist, X=y_anchor, Y=y_aug_pos, scale=scale
        )

        return y_aug_pos, y_aug_pos[aug_neg_idx], aug_neg_weights

    def _shared_forward_step(
        self,
        batch: BatchType,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
        augment_data: bool = True,
    ) -> torch.Tensor:
        batch_data, batch_class_weights = batch
        row_mask = compute_row_mask(batch_data)
        setwise_dist, item_flow = batch_chamfer_distance(
            X=batch_data, row_mask=row_mask
        )

        #### REAL DATA ####
        # positive mining
        triplet_sampler = TripletSetSampler(setwise_dist)
        pos_idx = triplet_sampler.positive_sampling()

        # forward pass
        y_anchor = self._forward(batch_data)

        # negative sampling
        scale = self.hparams["sample_scale"]
        neg_idx, neg_weights = triplet_sampler.negative_sampling(
            X=y_anchor, pos_idx=pos_idx, scale=scale
        )

        y_pos = y_anchor[pos_idx]
        y_neg = y_anchor[neg_idx]

        if augment_data:
            y_aug_pos, y_aug_neg, aug_neg_weights = self._augmented_forward_step(
                batch=batch_data,
                y_anchor=y_anchor,
                pos_idx=pos_idx,
                item_flow=item_flow,
                row_mask=row_mask,
            )
        else:
            y_aug_pos = None
            y_aug_neg = None
            aug_neg_weights = None

        # 5. Compute loss and log
        loss: torch.Tensor = self.criterion(
            y_self=y_anchor,
            y_pos=y_pos,
            y_neg=y_neg,
            neg_weights=neg_weights,
            class_weights=batch_class_weights,
            y_aug_pos=y_aug_pos,
            y_aug_neg=y_aug_neg,
            aug_neg_weights=aug_neg_weights,
        )

        self._log_loss(loss, stage)

        return loss

    def training_step(self, train_batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._shared_forward_step(train_batch, batch_idx, "train")

    def validation_step(self, val_batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._shared_forward_step(
            val_batch, batch_idx, "val", augment_data=False
        )

    def test_step(self, test_batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._shared_forward_step(
            test_batch, batch_idx, "test", augment_data=False
        )

    # TODO: idk if more is needed
    def predict_step(
        self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self(batch[0])


# TODO: these aren't working well with lightning CLI
# return final output from decoder for genome level embeddings
class GenomeTransformer(_ProteinSetTransformer):
    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(X, **kwargs)


# return final layer of encoder for gene level embeddings that are
# informed by genome architecture and homology across viruses
class GeneTransformer(_ProteinSetTransformer):
    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model.encode(X, **kwargs)
        idx = self.model.final_encoder_layer_idx
        return out[idx].repr

    def protein_weights(self, X: torch.Tensor) -> AttentionSchema:
        # the first module in the decoder is the pooling multihead attention block
        # that will? weight the proteins in each genome?
        return self.model._decoder[0](X, return_weights=True)

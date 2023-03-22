from __future__ import annotations

from typing import Literal

import lightning as L
import torch
from torch import optim

from .distance import SetDistance
from .loss import AugmentedWeightedTripletLoss
from .sampling import PointSwapSampler
from pst.arch.model import SetTransformer
from pst.utils.mask import compute_row_mask


class _ProteinSetTransformer(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_heads: int,
        *,
        # optimizer
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        # point swap sampling
        sample_scale: float = 7.0,
        sample_rate: float = 0.5,
        # loss
        loss_alpha: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.RWMDistance = SetDistance()
        self.criterion = AugmentedWeightedTripletLoss(self.hparams["loss_alpha"])

        # TODO: prob not necessary to make this flexible? idk i could test deep sets but tranformer prob better
        self.model = SetTransformer(
            in_dim=in_dim,
            n_outputs=1,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            **kwargs,
        )

    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "You instantiate a concrete subclass like GenomeClassifier or GeneClassifier"
        )

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            betas=self.hparams["betas"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def _shared_eval(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        # 1. Compute relaxed word mover's distance
        rwmd, flow = self.RWMDistance.fit_transform(batch)

        # 2. Compute row mask
        # TODO: this is technically computed twice?
        # oh well I guess? bc I can't precompute the attn_mask here
        row_mask = compute_row_mask(batch)

        # 3. Point-swap sampling
        sampler = PointSwapSampler(
            emd=rwmd,
            batch=batch,
            flow=flow,
            row_mask=row_mask,
            sample_rate=self.hparams["sample_rate"],
            scale=self.hparams["sample_scale"],
        )
        triple_sample, aug_sample = sampler.sample()

        forward_kwargs = dict(return_weights=False, attn_mask=None)

        # 4. Forward pass with batch, pos/neg samples, and augmented data
        # to do triplet loss.
        # TODO: break this into functions
        y_self = self(batch, **forward_kwargs)
        y_pos = self(batch[triple_sample.idx[1]], **forward_kwargs)
        y_neg = self(batch[triple_sample.idx[2]], **forward_kwargs)
        y_aug_pos = self(aug_sample.data, **forward_kwargs)
        y_aug_neg = self(aug_sample.data[aug_sample.negative_idx], **forward_kwargs)

        # 5. Compute loss and log
        loss: torch.Tensor = self.criterion(
            y_self=y_self,
            y_pos=y_pos,
            y_neg=y_neg,
            negative_weights=triple_sample.weights,
            y_aug_pos=y_aug_pos,
            y_aug_neg=y_aug_neg,
            aug_neg_weights=aug_sample.weights,
        )
        self.log(
            f"{stage}_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(train_batch, batch_idx, "train")

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(val_batch, batch_idx, "val")

    def test_step(self, test_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(test_batch, batch_idx, "test")

    # TODO: idk if more is needed
    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self(batch)


# return final output from decoder for genome level embeddings
class GenomeClassifier(_ProteinSetTransformer):
    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(X, **kwargs)


# return final layer of encoder for gene level embeddings that are
# informed by genome architecture and homology across viruses
class GeneClassifier(_ProteinSetTransformer):
    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model.encode(X, **kwargs)
        idx = self.model.final_encoder_layer_idx
        return out[idx].repr

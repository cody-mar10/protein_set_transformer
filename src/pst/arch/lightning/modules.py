from __future__ import annotations

from typing import Any, Literal, Optional

import lightning as L
import torch
from torch import optim

from .distance import SetDistance
from .loss import AugmentedWeightedTripletLoss
from .sampling import PointSwapSampler, PrecomputedSampling, PrecomputeSampler
from pst.arch.model import SetTransformer
from pst.utils.mask import compute_row_mask
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
        patience: int = 5,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        use_scheduler: bool = True,
        # point swap sampling
        sample_scale: float = 7.0,
        sample_rate: float = 0.5,
        # loss
        loss_alpha: float = 0.1,
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
                optimizer (OptimizerCallable, optional): type of optimizer. Defaults to torch.optim.Adam.
                scheduler (LRSchedulerCallable, optional): type of learning rate scheduler. Defaults to torch.optim.lr_scheduler.ConstantLR.
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

        self.RWMDistance = SetDistance()
        self.criterion = AugmentedWeightedTripletLoss(loss_alpha)
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
            norm=norm,
        )
        self.precomputed_sampling: Optional[PrecomputedSampling] = None

    def on_train_start(self) -> None:
        super().on_train_start()
        # attach precomputed results for model to access during training_step
        # datamodule.prepare_data() called before this, so it should be available
        file = self.trainer.datamodule.precomputed_sampling_file  # type: ignore
        self.precomputed_sampling = PrecomputeSampler.load_precomputed_sampling(file)

    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        # return self.model(X, **kwargs)
        raise NotImplementedError(
            "You instantiate a concrete subclass like GenomeClassifier or GeneClassifier"
        )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            betas=self.hparams["betas"],
            weight_decay=self.hparams["weight_decay"],
        )
        config: dict[str, Any] = {"optimizer": optimizer}
        if self.hparams["use_scheduler"]:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.hparams["patience"]
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            }
        return config

    def _shared_eval(
        self,
        batch: BatchType,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        batch_idx, batch_data = batch
        if self.precomputed_sampling is None:
            # 1. Compute relaxed word mover's distance
            rwmd, flow = self.RWMDistance.fit_transform(batch_data)

            # 2. Compute row mask
            # TODO: this is technically computed twice?
            # oh well I guess? bc I can't precompute the attn_mask here
            row_mask = compute_row_mask(batch_data, unsqueeze=False)

            # 3. Point-swap sampling
            # TODO: this is technically always the same each time since it doesn't consider model inputs...
            # meaning that this can be precomputed or cached
            # actually to be flexible with batch sizes, prob just compute upfront each time?
            sampler = PointSwapSampler(
                emd=rwmd,
                batch=batch_data,
                flow=flow,
                row_mask=row_mask,
                sample_rate=self.hparams["sample_rate"],
                scale=self.hparams["sample_scale"],
            )
            triplet_sample, aug_sample = sampler.sample()
            pos_idx = triplet_sample.idx[1]
            neg_idx = triplet_sample.idx[2]
            triplet_weights = triplet_sample.weights
            aug_data = aug_sample.data
            aug_neg_weights = aug_sample.weights
            aug_neg_idx = aug_sample.negative_idx
        else:
            device = batch_data.device
            triplet_sample = self.precomputed_sampling["triplet"]
            aug_sample = self.precomputed_sampling["aug"]

            # TODO: val batch idx restart from 0, I think sampler can return this
            pos_idx: torch.Tensor = triplet_sample["indices"][batch_idx][1].to(
                device=device
            )
            neg_idx: torch.Tensor = triplet_sample["indices"][batch_idx][2].to(
                device=device
            )
            triplet_weights: torch.Tensor = triplet_sample["weights"][batch_idx].to(
                device=device
            )
            aug_data: torch.Tensor = aug_sample["data"][batch_idx].to(device=device)
            aug_neg_weights: torch.Tensor = aug_sample["weights"][batch_idx].to(
                device=device
            )
            aug_neg_idx: torch.Tensor = aug_sample["negative_indices"][batch_idx].to(
                device=device
            )

        forward_kwargs = dict(return_weights=False, attn_mask=None)

        # 4. Forward pass with batch, pos/neg samples, and augmented data
        # to do triplet loss.
        # TODO: break this into functions
        y_self = self(batch_data, **forward_kwargs)
        y_pos = self(batch_data[pos_idx], **forward_kwargs)
        y_neg = self(batch_data[neg_idx], **forward_kwargs)
        y_aug_pos = self(aug_data, **forward_kwargs)
        y_aug_neg = self(aug_data[aug_neg_idx], **forward_kwargs)

        # 5. Compute loss and log
        loss: torch.Tensor = self.criterion(
            y_self=y_self,
            y_pos=y_pos,
            y_neg=y_neg,
            neg_weights=triplet_weights,
            y_aug_pos=y_aug_pos,
            y_aug_neg=y_aug_neg,
            aug_neg_weights=aug_neg_weights,
        )
        self.log(
            f"{stage}_loss",
            value=loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, train_batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(train_batch, batch_idx, "train")

    def validation_step(self, val_batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(val_batch, batch_idx, "val")

    def test_step(self, test_batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(test_batch, batch_idx, "test")

    # TODO: idk if more is needed
    def predict_step(
        self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self(batch[1])


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

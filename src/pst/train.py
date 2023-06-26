from __future__ import annotations

from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    Timer,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from pst.arch import EdgeIndexStrategy, GenomeDataModule, ProteinSetTransformer
from pst.cross_validation import CrossValEventSummarizer, CVStatusLogger
from pst.utils.cli import (
    _KWARG_TYPE,
    AcceleratorOpts,
    AnnealingOpts,
    Args,
    GradClipAlgOpts,
    PrecisionOpts,
    StrategyOpts,
)


class Trainer:
    # TODO: all defaults can be gotten from cli.py technically
    def __init__(
        self,
        file: Path,
        # model kwargs
        out_dim: int = -1,
        num_heads: int = 4,
        n_enc_layers: int = 5,
        multiplier: float = 1.0,
        dropout: float = 0.5,
        compile: bool = False,
        # optimizer kwargs
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 0,
        # data kwargs
        batch_size: int = 32,
        train_on_full: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        edge_strategy: EdgeIndexStrategy = "chunked",
        chunk_size: int = 30,  # TODO: get from data.py
        threshold: int = -1,  # TODO: get from data.py
        log_inverse: bool = False,
        # trainer kwargs
        devices: int = 1,
        accelerator: AcceleratorOpts = "gpu",
        default_root_dir: Path = Path("lightning_root"),
        max_epochs: int = 1000,
        precision: PrecisionOpts = "16-mixed",
        strategy: StrategyOpts = "ddp",
        gradient_clip_algorithm: Optional[GradClipAlgOpts] = None,
        gradient_clip_val: Optional[float] = None,
        max_time: Optional[timedelta] = None,
        # experiment kwargs
        name: str = "exp0",
        patience: int = 5,
        save_top_k: int = 3,
        swa: bool = False,
        swa_epoch_start: int | float = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: AnnealingOpts = "linear",
        # loss kwargs
        margin: float = 0.1,
        # augmentation kwargs
        sample_scale: float = 7.0,
        sample_rate: float = 0.5,
        **trainer_kwargs,
    ) -> None:
        self.data_kwargs: _KWARG_TYPE = dict(
            file=file,
            batch_size=batch_size,
            train_on_full=train_on_full,
            num_workers=num_workers,
            pin_memory=pin_memory,
            edge_strategy=edge_strategy,
            chunk_size=chunk_size,
            threshold=threshold,
            log_inverse=log_inverse,
        )
        self.load_datamodule(**self.data_kwargs)

        self.loss_kwargs = {"margin": margin}
        self.augmentation_kwargs = dict(
            sample_rate=sample_rate,
            sample_scale=sample_scale,
        )
        self.optimizer_kwargs: _KWARG_TYPE = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            warmup_steps=warmup_steps,
        )

        self.model_kwargs: _KWARG_TYPE = dict(
            in_dim=self.datamodule.dataset.feature_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            n_enc_layers=n_enc_layers,
            multiplier=multiplier,
            dropout=dropout,
            compile=compile,
            optimizer_kwargs=self.optimizer_kwargs,
            loss_kwargs=self.loss_kwargs,
            augmentation_kwargs=self.augmentation_kwargs,
        )

        self.trainer_kwargs: _KWARG_TYPE = dict(
            devices=devices,
            accelerator=accelerator,
            default_root_dir=default_root_dir,
            max_epochs=max_epochs,
            precision=precision,
            strategy=strategy,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            **trainer_kwargs,
        )
        # keep separately to use as a callback
        self.max_time = max_time

        self.experiment_kwargs: _KWARG_TYPE = dict(
            name=name,
            patience=patience,
            save_top_k=save_top_k,
            swa=swa,
            swa_kwargs=dict(
                swa_epoch_start=swa_epoch_start,
                annealing_epochs=annealing_epochs,
                annealing_strategy=annealing_strategy,
            ),
        )

    @classmethod
    def from_cli_args(cls, args: Args):
        kwargs = {
            key: value for f in fields(args) for key, value in getattr(args, f.name)
        }
        kwargs.pop("predict")
        kwargs.pop("mode")

        return cls(**kwargs)

    @classmethod
    def from_kwargs(
        cls,
        model_kwargs: _KWARG_TYPE,
        data_kwargs: _KWARG_TYPE,
        optimizer_kwargs: _KWARG_TYPE,
        loss_kwargs: _KWARG_TYPE,
        augmentation_kwargs: _KWARG_TYPE,
        trainer_kwargs: _KWARG_TYPE,
        experiment_kwargs: _KWARG_TYPE,
    ):
        kwargs = (
            model_kwargs
            | data_kwargs
            | optimizer_kwargs
            | loss_kwargs
            | augmentation_kwargs
            | trainer_kwargs
            | experiment_kwargs
        )
        return cls(**kwargs)

    def trainer_callbacks(self, num_folds: Optional[int] = None) -> list[L.Callback]:
        callbacks: list[L.Callback] = list()

        if num_folds is not None and self.max_time is not None:
            max_time = self.max_time / num_folds
        else:
            max_time = self.max_time

        callbacks.append(Timer(duration=max_time))

        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=self.experiment_kwargs["save_top_k"],
                every_n_epochs=1,
                save_last=True,
                filename="{epoch}-{step}-{val_loss}",
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=self.experiment_kwargs["patience"],
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        if self.experiment_kwargs["swa"]:
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=self.optimizer_kwargs["lr"],
                    **self.experiment_kwargs["swa_kwargs"],
                )
            )

        return callbacks

    def load_datamodule(self, **data_kwargs):
        self.datamodule = GenomeDataModule(**data_kwargs)
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

    def train(
        self,
        trainer: L.Trainer,
        model: L.LightningModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    def train_with_cross_validation(self):
        # will only apply shuffling to training dataloader and NOT val dataloader
        dataloaders = self.datamodule.train_val_dataloaders(shuffle=True)

        for fold_idx, cv_dataloader in enumerate(dataloaders):
            train_loader = cv_dataloader.train_loader
            val_loader = cv_dataloader.val_loader
            train_group_ids = cv_dataloader.train_group_ids
            val_group_id = cv_dataloader.val_group_id

            enable_model_summary = fold_idx == 0
            model = ProteinSetTransformer(**self.model_kwargs)

            logger = TensorBoardLogger(
                save_dir=self.trainer_kwargs["default_root_dir"],
                name=self.experiment_kwargs["name"],
                version=fold_idx,
                default_hp_metric=False,
            )
            callbacks = self.trainer_callbacks()
            callbacks.append(
                CVStatusLogger(
                    fold_idx=fold_idx,
                    train_group_ids=train_group_ids,
                    val_group_id=val_group_id,
                )
            )
            trainer = L.Trainer(
                callbacks=callbacks,
                logger=logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                enable_model_summary=enable_model_summary,
                **self.trainer_kwargs,
            )
            self.train(
                trainer=trainer,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
            )

        exp_name = self.experiment_kwargs["name"]
        print(
            f"Cross validation experiment {exp_name} finished. "
            f"Summarizing validation loss per epoch per fold in log dir."
        )
        self.summarize_cross_validation()

    def summarize_cross_validation(self):
        cv_summarizer = CrossValEventSummarizer(
            root_dir=self.trainer_kwargs["default_root_dir"],
            name=self.experiment_kwargs["name"],
        )
        # TODO: add in args
        cv_summarizer.summarize_and_save(
            output_name="cross_validation_summary.tsv",
            metric="val_loss",
            step="epoch",
        )

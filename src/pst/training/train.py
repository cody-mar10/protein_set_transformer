from __future__ import annotations

from typing import Any

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from pst.arch import GenomeDataModule, ProteinSetTransformer
from pst.training import CrossValEventSummarizer
from pst.utils.cli import Args


class Trainer:
    def __init__(
        self,
        model_kwargs: dict[str, Any],
        optimizer_kwargs: dict[str, Any],
        data_kwargs: dict[str, Any],
        trainer_kwargs: dict[str, Any],
        experiment_kwargs: dict[str, Any],
        loss_kwargs: dict[str, Any],
        augmentation_kwargs: dict[str, Any],
    ) -> None:
        self.model_kwargs = model_kwargs | optimizer_kwargs | augmentation_kwargs
        self.data_kwargs = data_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.experiment_kwargs = experiment_kwargs
        self.loss_kwargs = loss_kwargs

    @classmethod
    def from_cli_args(cls, args: Args):
        return cls(
            model_kwargs=args.model,
            optimizer_kwargs=args.optimizer,
            data_kwargs=args.data,
            trainer_kwargs=args.trainer,
            experiment_kwargs=args.experiment,
            loss_kwargs=args.loss,
            augmentation_kwargs=args.augmentation,
        )

    # TODO: make configurable from cli
    def trainer_callbacks(self) -> list[L.Callback]:
        callbacks: list[L.Callback] = list()

        callbacks.append(
            ModelCheckpoint(monitor="val_loss", save_top_k=10, every_n_epochs=1)
        )
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=self.experiment_kwargs["patience"],
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=self.model_kwargs["lr"],
                swa_epoch_start=0.75,
                annealing_strategy="linear",
            )
        )

        return callbacks

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
        datamodule = GenomeDataModule(**self.data_kwargs)
        datamodule.prepare_data()
        datamodule.setup("fit")

        dataloaders = datamodule.train_val_dataloaders(shuffle=True)

        for fold_idx, (train_loader, val_loader) in enumerate(dataloaders):
            model = ProteinSetTransformer(**self.model_kwargs)

            logger = TensorBoardLogger(
                save_dir=self.trainer_kwargs["default_root_dir"],
                name=self.experiment_kwargs["name"],
                version=fold_idx,
                default_hp_metric=False,
            )
            trainer = L.Trainer(
                callbacks=self.trainer_callbacks(),
                logger=logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                **self.trainer_kwargs,
            )
            self.train(
                trainer=trainer,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
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

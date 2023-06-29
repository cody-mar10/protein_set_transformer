from __future__ import annotations

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
from optuna import TrialPruned
from torch.utils.data import DataLoader

from pst.arch import EdgeIndexStrategy, GenomeDataModule, ProteinSetTransformer
from pst.cross_validation import CrossValEventSummarizer, CVStatusLogger
from pst.utils.cli import (
    _KWARG_TYPE,
    NO_NEGATIVES_MODES,
    AcceleratorOpts,
    AnnealingOpts,
    Args,
    GradClipAlgOpts,
    PrecisionOpts,
    StrategyOpts,
)


class CrossValidationTrainer:
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
        limit_train_batches: Optional[int | float] = None,
        limit_val_batches: Optional[int | float] = None,
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
        no_negatives_mode: NO_NEGATIVES_MODES = "closest_to_positive",
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

        self.loss_kwargs = {"margin": margin}
        self.augmentation_kwargs = dict(
            sample_rate=sample_rate,
            sample_scale=sample_scale,
            no_negatives_mode=no_negatives_mode,
        )
        self.optimizer_kwargs: _KWARG_TYPE = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            warmup_steps=warmup_steps,
        )

        self.model_kwargs: _KWARG_TYPE = dict(
            # in_dim=self.datamodule.dataset.feature_dim,
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
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
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

        # hold callback constructors to be created each fold
        self._callbacks: list[tuple[type[L.Callback], _KWARG_TYPE]] = list()
        # populate with a default set of callbacks
        self.default_callbacks()

    @classmethod
    def from_cli_args(cls, args: Args):
        instance = cls.from_kwargs(
            model_kwargs=args.model,
            data_kwargs=args.data,
            optimizer_kwargs=args.optimizer,
            loss_kwargs=args.loss,
            augmentation_kwargs=args.augmentation,
            trainer_kwargs=args.trainer,
            experiment_kwargs=args.experiment,
        )

        return instance

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

    def register_callback(self, callback: type[L.Callback], **kwargs):
        callback_constructor = (callback, kwargs)
        self._callbacks.append(callback_constructor)

    def default_callbacks(self):
        self.register_callback(
            ModelCheckpoint,
            monitor="val_loss",
            save_top_k=self.experiment_kwargs["save_top_k"],
            every_n_epochs=1,
            save_last=True,
            filename="{epoch}-{step}-{val_loss:.3f}",
        )
        self.register_callback(LearningRateMonitor, logging_interval="step")

        # TODO: make this optional
        # TODO: add tuning arg
        # actual prob don't need this
        if self.experiment_kwargs.get("early_stopping", False):
            self.register_callback(
                EarlyStopping,
                monitor="val_loss",
                min_delta=0.0,
                patience=self.experiment_kwargs["patience"],
            )

        if self.experiment_kwargs["swa"]:
            self.register_callback(
                StochasticWeightAveraging,
                swa_lrs=self.optimizer_kwargs["lr"],
                **self.experiment_kwargs["swa_kwargs"],
            )

    def trainer_callbacks(self) -> list[L.Callback]:
        callbacks = [
            callback(**kwargs) for callback, kwargs in self._callbacks  # type: ignore
        ]

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

    def on_train_with_cross_validation_start(self):
        # TODO: maybe register hyperparams?
        self.load_datamodule(**self.data_kwargs)
        self._cv_scores: list[float] = list()

    def on_fold_training_start(self):
        pass

    def on_before_trainer_init(
        self,
        callbacks: list[L.Callback],
        fold_idx: int,
        train_group_ids: list[int],
        val_group_id: int,
    ):
        cv_status_logger_callback = CVStatusLogger(
            fold_idx=fold_idx,
            train_group_ids=train_group_ids,
            val_group_id=val_group_id,
        )
        self.add_callback_during_fold(callbacks, cv_status_logger_callback)

    def on_after_trainer_init(self, trainer: L.Trainer):
        """Called immediately before the actual training loop is called"""
        pass

    def on_fold_training_end(self, trainer: L.Trainer):
        # this is the val loss of the final epoch, maybe should look into other epocs,
        # ie best epoch
        val_loss: float = trainer.callback_metrics["val_loss"].item()
        self._cv_scores.append(val_loss)

    def on_train_with_cross_validation_end(self, was_pruned: bool) -> float:
        if not was_pruned:
            exp_name = self.experiment_kwargs["name"]
            print(
                f"Cross validation experiment {exp_name} finished. "
                f"Summarizing validation loss per epoch per fold in log dir."
            )
            self.summarize_cross_validation()
        else:
            # don't try to summarize if trial is pruned
            print("Trial was pruned")
        avg_val_loss: float = float("inf")
        if self._cv_scores:
            avg_val_loss = sum(self._cv_scores) / len(self._cv_scores)
        return avg_val_loss

    def add_callback_during_fold(
        self, callbacks: list[L.Callback], addition: L.Callback
    ):
        """Add a `lightning.Callback` instance during the CV-training loop.

        Args:
            callbacks (list[L.Callback]): list of `lightning.Callback` instances
            addition (L.Callback): new `L.Callback` instance.
        """
        callbacks.append(addition)

    def train_with_cross_validation(self) -> float:
        self.on_train_with_cross_validation_start()
        # will only apply shuffling to training dataloader and NOT val dataloader
        dataloaders = self.datamodule.train_val_dataloaders(shuffle=True)
        num_folds = self.datamodule.data_manager.n_folds

        # adjust max time since that is max time for a single fold
        if self.max_time is not None:
            duration = self.max_time / num_folds
            self.register_callback(Timer, duration=duration)

        in_dim = self.datamodule.dataset.feature_dim

        was_pruned = False
        for fold_idx, cv_dataloader in enumerate(dataloaders):
            self.on_fold_training_start()

            train_loader = cv_dataloader.train_loader
            val_loader = cv_dataloader.val_loader
            train_group_ids = cv_dataloader.train_group_ids
            val_group_id = cv_dataloader.val_group_id

            enable_model_summary = fold_idx == 0
            model = ProteinSetTransformer(in_dim=in_dim, **self.model_kwargs)

            logger = TensorBoardLogger(
                save_dir=self.trainer_kwargs["default_root_dir"],
                name=self.experiment_kwargs["name"],
                version=fold_idx,
                default_hp_metric=False,
            )

            callbacks = self.trainer_callbacks()
            self.on_before_trainer_init(
                callbacks=callbacks,
                fold_idx=fold_idx,
                train_group_ids=train_group_ids,
                val_group_id=val_group_id,
            )
            trainer = L.Trainer(
                callbacks=callbacks,
                logger=logger,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                enable_model_summary=enable_model_summary,
                **self.trainer_kwargs,
            )

            self.on_after_trainer_init(trainer)
            try:
                self.train(
                    trainer=trainer,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )
            except TrialPruned:
                # if one fold is pruned, just cancel entire thing
                # since we want a good trial for all folds
                was_pruned = True
                break
            self.on_fold_training_end(trainer)
        return self.on_train_with_cross_validation_end(was_pruned)

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

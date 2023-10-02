from __future__ import annotations

import lightning_cv as lcv
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning_cv.callbacks.base import Callback
from lightning_cv.callbacks.checkpoint import ModelCheckpoint
from lightning_cv.callbacks.lr_monitor import LearningRateMonitor

from pst.data.modules import GenomeDataModule
from pst.nn.modules import CrossValPST as PST
from pst.utils.cli.modes import TrainingMode


def train_with_cross_validation(config: TrainingMode):
    trainer_callbacks: list[Callback] = [
        ModelCheckpoint(
            save_last=True, save_top_k=config.experiment.save_top_k, every_n_epochs=1
        )
    ]

    if config.model.optimizer.use_scheduler:
        trainer_callbacks.append(LearningRateMonitor())

    datamodule = GenomeDataModule(config.data)
    # update model's in_dim
    if config.model.in_dim == -1:
        config.model.in_dim = datamodule.dataset.feature_dim

    logger = CSVLogger(
        root_dir=config.trainer.default_root_dir, name=config.experiment.name
    )
    trainer_config = lcv.CrossValidationTrainerConfig(
        loggers=logger,
        callbacks=trainer_callbacks,
        limit_train_batches=config.trainer.limit_train_batches or 1.0,
        limit_val_batches=config.trainer.limit_val_batches or 1.0,
        checkpoint_dir=config.trainer.default_root_dir,
        # everything else is the same name
        **config.trainer.model_dump(
            include={"accelerator", "strategy", "devices", "precision", "max_epochs"}
        ),
    )

    trainer = lcv.CrossValidationTrainer(
        model_type=PST,  # type: ignore
        config=trainer_config,
    )

    # TODO: need to figure out how to update positional embedding max size if needed
    # simplest solution is to just pass that as an arg to the model thru the config...
    trainer.train_with_cross_validation(
        datamodule=datamodule,
        model_config=config.model,
    )

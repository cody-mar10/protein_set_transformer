from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from pst.training.debug import LGradientLogger
from pst.training.utils.constants import (
    CHECKPOINT_EVERY_N_EPOCHS,
    MIN_DELTA,
    STOPPING_THRESHOLD,
)
from pst.utils.cli.modes import TrainingMode


def get_callbacks(
    config: TrainingMode, checkpoint: bool = True, early_stopping: bool = True
) -> list[Callback]:
    callbacks: list[Callback] = list()

    if checkpoint:
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch}_{train_loss:.3f}",
                save_last=True,
                save_top_k=config.experiment.save_top_k,
                every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
                monitor="train_loss",
                mode="min",
            )
        )

    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                patience=config.experiment.patience,
                verbose=True,
                mode="min",
                check_finite=True,
                strict=True,
                stopping_threshold=STOPPING_THRESHOLD,
                min_delta=MIN_DELTA,
            )
        )

    return callbacks


def init_lightning_trainer(
    config: TrainingMode,
    checkpoint: bool = True,
    early_stopping: bool = True,
) -> L.Trainer:
    callbacks = get_callbacks(
        config, checkpoint=checkpoint, early_stopping=early_stopping
    )

    if config.experiment.debug:
        callbacks.append(LGradientLogger())

    trainer = L.Trainer(
        # want max time value, not enum
        max_time=config.trainer.max_time.value,
        callbacks=callbacks,
        **config.trainer.model_dump(exclude={"max_time"}),
        detect_anomaly=config.experiment.debug,
    )
    return trainer

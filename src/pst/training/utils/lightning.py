import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from pst.nn.config import BaseModelConfig
from pst.training.utils.constants import (
    CHECKPOINT_EVERY_N_EPOCHS,
    MIN_DELTA,
    STOPPING_THRESHOLD,
)
from pst.utils.cli.trainer import TrainerArgs


def get_callbacks(
    trainer_cfg: TrainerArgs,
    checkpoint: bool = True,
    early_stopping: bool = True,
    monitor: str = "train_loss",
) -> list[Callback]:
    callbacks: list[Callback] = list()

    if checkpoint:
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch}_{train_loss:.3f}",
                save_last=True,
                save_top_k=trainer_cfg.extra.save_top_k,
                every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
                monitor=monitor,
                mode="min",
            )
        )

    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                patience=trainer_cfg.extra.patience,
                verbose=True,
                mode="min",
                check_finite=True,
                strict=True,
                stopping_threshold=STOPPING_THRESHOLD,
                min_delta=trainer_cfg.extra.min_delta or MIN_DELTA,
            )
        )

    return callbacks


def init_lightning_trainer(
    model_cfg: BaseModelConfig,
    trainer_cfg: TrainerArgs,
    checkpoint: bool = True,
    early_stopping: bool = True,
    monitor: str = "train_loss",
) -> L.Trainer:
    callbacks = get_callbacks(
        trainer_cfg=trainer_cfg,
        checkpoint=checkpoint,
        early_stopping=early_stopping,
        monitor=monitor,
    )

    if model_cfg.optimizer.use_scheduler:
        callbacks.append(
            LearningRateMonitor(logging_interval="epoch", log_momentum=True)
        )

    trainer = L.Trainer(
        # want max time value, not enum
        max_time=trainer_cfg.max_time.value,
        precision=trainer_cfg.precision.value,
        callbacks=callbacks,
        **trainer_cfg.to_dict(exclude={"max_time", "extra", "precision"}),
    )
    return trainer

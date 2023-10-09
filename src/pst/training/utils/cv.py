import logging
from typing import Optional

from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.logger import Logger
from lightning_cv import CrossValidationTrainerConfig
from lightning_cv.callbacks.base import Callback as Callback
from lightning_cv.callbacks.checkpoint import ModelCheckpoint
from lightning_cv.callbacks.lr_monitor import LearningRateMonitor
from lightning_cv.callbacks.stopping import EarlyStopping
from lightning_cv.callbacks.timer import Timer

from pst.training.debug import debugify
from pst.training.utils.constants import (
    BUFFER,
    CHECKPOINT_EVERY_N_EPOCHS,
    MIN_DELTA,
    STOPPING_THRESHOLD,
    TIMER_STOP_INTERVAL,
)
from pst.utils.cli.modes import TrainingMode
from pst.utils.cli.trainer import TrainerArgs

log = logging.getLogger(__name__)


def get_callbacks(
    config: TrainingMode,
    checkpoint: bool = True,
    early_stopping: bool = True,
    timer: bool = True,
) -> list[Callback]:
    callbacks: list[Callback] = list()

    if checkpoint:
        callbacks.append(
            ModelCheckpoint(
                save_last=True,
                save_top_k=config.experiment.save_top_k,
                every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
            )
        )

    if early_stopping:
        callbacks.append(
            EarlyStopping(
                patience=config.experiment.patience,
                verbose=True,
                mode="min",
                check_finite=True,
                stopping_threshold=STOPPING_THRESHOLD,
                min_delta=MIN_DELTA,
            )
        )

    if timer:
        callbacks.append(
            Timer(
                duration=config.trainer.max_time.value,
                buffer=BUFFER,
                interval=TIMER_STOP_INTERVAL,
            )
        )

    return callbacks


def get_trainer_config(
    trainer_args: TrainerArgs,
    callbacks: list[Callback],
    logger: Optional[Logger] = None,
) -> CrossValidationTrainerConfig:
    config = CrossValidationTrainerConfig(
        loggers=logger,
        callbacks=callbacks,
        limit_train_batches=trainer_args.limit_train_batches or 1.0,
        limit_val_batches=trainer_args.limit_val_batches or 1.0,
        checkpoint_dir=trainer_args.default_root_dir,
        **trainer_args.model_dump(
            include={
                "accelerator",
                "strategy",
                "devices",
                "precision",
                "max_epochs",
                "gradient_clipping_algorithm",
                "gradient_clip_val",
            }
        ),
    )
    return config


def init_trainer_config(
    config: TrainingMode,
    checkpoint: bool = True,
    early_stopping: bool = True,
    timer: bool = True,
    add_logger: bool = True,
) -> CrossValidationTrainerConfig:
    callbacks = get_callbacks(
        config,
        checkpoint,
        early_stopping,
        timer,
    )

    if config.model.optimizer.use_scheduler:
        callbacks.append(LearningRateMonitor())

    if add_logger:
        logger = CSVLogger(
            root_dir=config.trainer.default_root_dir,
            name=config.experiment.name,
        )
    else:
        logger = None

    trainer_config = get_trainer_config(
        config.trainer,
        callbacks,
        logger,
    )

    if config.experiment.log_gradients:
        log.warning(
            "Logging gradients is on, and will be written to disk, which may take up a "
            "lot of space."
        )
        debugify(trainer_config, config.trainer)

    return trainer_config

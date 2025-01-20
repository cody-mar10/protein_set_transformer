import logging
from typing import Optional

from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.logger import Logger
from lightning_cv import CrossValidationTrainerConfig
from lightning_cv.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)

from pst.nn.config import BaseModelConfig
from pst.training.utils.constants import (
    BUFFER,
    CHECKPOINT_EVERY_N_EPOCHS,
    MIN_DELTA,
    STOPPING_THRESHOLD,
    TIMER_STOP_INTERVAL,
)
from pst.utils.cli.experiment import ExperimentArgs
from pst.utils.cli.trainer import TrainerArgs

log = logging.getLogger(__name__)


def get_callbacks(
    trainer_cfg: TrainerArgs,
    checkpoint: bool = True,
    early_stopping: bool = True,
    timer: bool = True,
) -> list[Callback]:
    callbacks: list[Callback] = list()

    if checkpoint:
        callbacks.append(
            ModelCheckpoint(
                save_last=True,
                save_top_k=trainer_cfg.extra.save_top_k,
                every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
            )
        )

    if early_stopping:
        callbacks.append(
            EarlyStopping(
                patience=trainer_cfg.extra.patience,
                verbose=True,
                mode="min",
                check_finite=True,
                stopping_threshold=STOPPING_THRESHOLD,
                min_delta=trainer_cfg.extra.min_delta or MIN_DELTA,
            )
        )

    if timer:
        callbacks.append(
            Timer(
                duration=trainer_cfg.max_time.value,
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
        grad_accum_steps=trainer_args.accumulate_grad_batches,
        precision=trainer_args.precision.value,
        accelerator=trainer_args.accelerator.value,
        strategy=trainer_args.strategy.value,
        **trainer_args.to_dict(
            include={
                "devices",
                "max_epochs",
                "gradient_clipping_algorithm",
                "gradient_clip_val",
            }
        ),
    )
    return config


def init_trainer_config(
    model_cfg: BaseModelConfig,
    experiment: ExperimentArgs,
    trainer_cfg: TrainerArgs,
    checkpoint: bool = True,
    early_stopping: bool = True,
    timer: bool = True,
    add_logger: bool = True,
) -> CrossValidationTrainerConfig:
    callbacks = get_callbacks(
        trainer_cfg,
        checkpoint,
        early_stopping,
        timer,
    )

    if model_cfg.optimizer.use_scheduler:
        callbacks.append(LearningRateMonitor())

    if add_logger:
        logger = CSVLogger(
            root_dir=trainer_cfg.default_root_dir,
            name=experiment.name,
        )
    else:
        logger = None

    trainer_config = get_trainer_config(
        trainer_cfg,
        callbacks,
        logger,
    )

    return trainer_config

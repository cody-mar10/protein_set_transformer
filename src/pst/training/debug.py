from typing import cast

import lightning as L
import lightning_cv as lcv
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities.grads import grad_norm
from lightning_cv import CrossValidationTrainerConfig
from lightning_cv.callbacks.base import Callback
from lightning_cv.typehints import KwargType
from torch.optim import Optimizer

from pst.utils.cli.trainer import TrainerArgs


class LGradientLogger(L.Callback):
    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:
        norms = grad_norm(pl_module, norm_type=2)
        pl_module.log_dict(
            norms,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            logger=True,
        )


class LcvGradientLogger(Callback):
    STAGE_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(self, log_pre_clip: bool = True, log_post_clip: bool = True):
        self.log_pre_clip = log_pre_clip
        self.log_post_clip = log_post_clip

    def log_grad(self, trainer: lcv.CrossValidationTrainer, prefix: str):
        current_fold = trainer.current_fold
        current_step = trainer.global_step_per_fold[current_fold]
        model = cast(
            L.LightningModule,
            trainer.fold_manager[current_fold].model,
        )

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(
                    f"{prefix}grad/{current_fold}/{name}", param.grad, current_step
                )

    def on_train_start(self, trainer: lcv.CrossValidationTrainer):
        if self.log_pre_clip or self.log_post_clip:
            for logger in trainer.fabric.loggers:
                if isinstance(logger, TensorBoardLogger):
                    self.logger = logger
                    break
            else:
                raise RuntimeError(
                    "No TensorBoardLogger found in trainer.fabric.loggers"
                )

    def on_after_backward(self, trainer: lcv.CrossValidationTrainer):
        if self.log_pre_clip:
            self.log_grad(trainer, "pre_clip_")

    def on_before_zero_grad(
        self, trainer: lcv.CrossValidationTrainer, optimizer: Optimizer
    ):
        if self.log_post_clip:
            self.log_grad(trainer, "post_clip_")

    def on_before_log_metrics(
        self,
        trainer: lcv.CrossValidationTrainer,
        metrics: KwargType,
        optimizer: Optimizer,
    ):
        # tensorboard logger doesn't like str values
        stage: str = metrics["stage"]
        metrics["stage"] = LcvGradientLogger.STAGE_MAP[stage]


def debugify(cv_config: CrossValidationTrainerConfig, trainer_args: TrainerArgs):
    if cv_config.loggers is None:
        cv_config.loggers = [
            CSVLogger(root_dir=trainer_args.default_root_dir, name="debug")
        ]
    elif not isinstance(cv_config.loggers, list):
        cv_config.loggers = [cv_config.loggers]

    cv_config.loggers.append(
        TensorBoardLogger(root_dir=trainer_args.default_root_dir, name="debug")
    )

    if cv_config.callbacks is not None:
        if not isinstance(cv_config.callbacks, list):
            cv_config.callbacks = [cv_config.callbacks]

        cv_config.callbacks.append(LcvGradientLogger())

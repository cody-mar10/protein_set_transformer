from __future__ import annotations

from functools import partial

import lightning_cv as lcv
import optuna

from pst.arch import GenomeDataModule
from pst.arch.modules import CrossValPST as PST
from pst.utils.cli import TrainingMode

from .optuna import OptunaIntegration
from .utils import _peek_feature_dim


def _optimize(trial: optuna.Trial, tuner: lcv.tuning.Tuner) -> float:
    return tuner.tune(trial=trial)


def tune(config: TrainingMode):
    # update model's in_dim
    if config.model.in_dim == -1:
        config.model.in_dim = _peek_feature_dim(config.data.file)

    ### CV TRAINER INIT
    trainer_callbacks: list[lcv.callbacks.Callback] = [
        lcv.callbacks.ModelCheckpoint(
            save_last=True, save_top_k=config.experiment.save_top_k, every_n_epochs=1
        ),
        lcv.callbacks.EarlyStopping(
            patience=config.experiment.patience,
            stopping_threshold=1e-3,  # checks loss < this, min loss is 0.0
        ),
    ]

    if config.model.optimizer.use_scheduler:
        trainer_callbacks.append(lcv.callbacks.LearningRateMonitor())

    trainer_config = lcv.CrossValidationTrainerConfig(
        loggers=None,  # defaults to CSVLogger THRU TUNER
        callbacks=trainer_callbacks,
        limit_train_batches=config.trainer.limit_train_batches or 1.0,
        limit_val_batches=config.trainer.limit_val_batches or 1.0,
        checkpoint_dir=config.trainer.default_root_dir,
        # everything else is the same name
        **config.trainer.model_dump(
            include={"accelerator", "strategy", "devices", "precision", "max_epochs"}
        ),
    )

    ### CV TUNER INIT
    tuner = lcv.tuning.Tuner(
        model_type=PST,  # type: ignore
        model_config=config.model,
        datamodule_type=GenomeDataModule,  # type: ignore
        datamodule_config=config.data,
        trainer_config=trainer_config,
        logdir=config.trainer.default_root_dir,
        experiment_name=config.experiment.name,
        hparam_config_file=config.experiment.config,
    )

    ### OPTUNA STUDY
    study = OptunaIntegration(
        expt_cfg=config.experiment, default_root_dir=config.trainer.default_root_dir
    ).study()

    optimize = partial(_optimize, tuner=tuner)
    optuna_callbacks = [
        lcv.tuning.callbacks.OptunaHyperparameterLogger(
            root_dir=config.trainer.default_root_dir,
            expt_name=config.experiment.name,
        )
    ]
    study.optimize(
        func=optimize,
        n_trials=config.experiment.n_trials,
        callbacks=optuna_callbacks,
        gc_after_trial=True,
    )

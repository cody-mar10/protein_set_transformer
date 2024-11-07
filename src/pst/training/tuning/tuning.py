from __future__ import annotations

import logging
from functools import partial

import lightning_cv as lcv
import optuna
from lightning_cv.tuning.callbacks import OptunaHyperparameterLogger

from pst.data.modules import GenomeDataModule
from pst.training.tuning.optuna import OptunaIntegration
from pst.training.utils.cv import init_trainer_config
from pst.training.utils.dim import check_feature_dim
from pst.utils.auto import auto_resolve_model_type
from pst.utils.cli.modes import TuningMode

logger = logging.getLogger(__name__)


def _optimize(trial: optuna.Trial, tuner: lcv.tuning.Tuner) -> float:
    return tuner.tune(trial=trial)


def tune(config: TuningMode):
    logger.info("Tuning hyperparameters with cross validation.")
    check_feature_dim(config)

    ### CV TRAINER INIT
    trainer_config = init_trainer_config(
        config,
        checkpoint=True,
        early_stopping=True,
        timer=True,
        add_logger=False,  # Tuner adds logger
    )

    model_type = auto_resolve_model_type(config.experiment.pst_model_type)

    ### CV TUNER INIT
    tuner = lcv.tuning.Tuner(
        model_type=model_type,  # type: ignore
        model_config=config.model,
        datamodule_type=GenomeDataModule,  # type: ignore
        datamodule_config=config.data,
        trainer_config=trainer_config,
        logdir=config.trainer.default_root_dir,
        experiment_name=config.experiment.name,
        hparam_config_file=config.experiment.config,
    )

    ### OPTUNA STUDY
    integration = OptunaIntegration(
        expt_name=tuner.experiment_name,
        default_root_dir=tuner.logdir,
        **config.tuning.model_dump(
            exclude={"config", "parallel"},
        ),
    )

    optimize = partial(_optimize, tuner=tuner)
    integration.register_callbacks(
        OptunaHyperparameterLogger(
            root_dir=config.trainer.default_root_dir,
            expt_name=config.experiment.name,
        )
    )

    integration.optimize(fn=optimize)

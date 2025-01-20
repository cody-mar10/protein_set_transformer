import logging
from functools import partial

import optuna
from lightning_cv.tuning import OptunaHyperparameterLogger, Tuner

from pst.data.config import CrossValDataConfig
from pst.data.modules import CrossValGenomeDataModule
from pst.nn.base import BaseModelTypes
from pst.nn.config import BaseModelConfig
from pst.nn.modules import ProteinSetTransformer
from pst.training.tuning.optuna import OptunaIntegration
from pst.training.utils.cv import init_trainer_config
from pst.training.utils.pst import _add_group_weights
from pst.utils.cli.experiment import ExperimentArgs
from pst.utils.cli.trainer import TrainerArgs
from pst.utils.cli.tuning import TuningArgs

logger = logging.getLogger(__name__)


class _PatchedTuner(Tuner):
    # this is to patch in the weights for PST for backwards compatibility

    def _create_datamodule(self):
        datamodule = super()._create_datamodule()

        if self.model_type is ProteinSetTransformer:
            _add_group_weights(datamodule)  # type: ignore

        return datamodule


def _optimize(trial: optuna.Trial, tuner: Tuner) -> float:
    return tuner.tune(trial=trial)


def tune(
    model_type: BaseModelTypes,
    model_cfg: BaseModelConfig,
    data: CrossValDataConfig,
    trainer_cfg: TrainerArgs,
    experiment: ExperimentArgs,
    tuning: TuningArgs,
):
    logger.info("Tuning hyperparameters with cross validation.")

    ### CV TRAINER INIT
    trainer_config = init_trainer_config(
        model_cfg,
        experiment,
        trainer_cfg,
        checkpoint=True,
        early_stopping=True,
        timer=True,
        add_logger=False,  # Tuner adds logger
    )

    ### CV TUNER INIT
    tuner = _PatchedTuner(
        model_type=model_type,
        model_config=model_cfg,
        datamodule_type=CrossValGenomeDataModule,
        datamodule_config=data,
        trainer_config=trainer_config,
        logdir=trainer_cfg.default_root_dir,
        experiment_name=experiment.name,
        hparam_config_file=tuning.config,
    )

    ### OPTUNA STUDY
    integration = OptunaIntegration(
        expt_name=tuner.experiment_name,
        default_root_dir=tuner.logdir,
        **tuning.to_dict(exclude={"config", "parallel"}),
    )

    optimize = partial(_optimize, tuner=tuner)
    integration.register_callbacks(
        OptunaHyperparameterLogger(
            root_dir=trainer_cfg.default_root_dir,
            expt_name=experiment.name,
        )
    )

    integration.optimize(fn=optimize)

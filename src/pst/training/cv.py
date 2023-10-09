from __future__ import annotations

import logging

import lightning_cv as lcv

from pst.data.modules import GenomeDataModule
from pst.nn.modules import CrossValPST as PST
from pst.training.utils.cv import init_trainer_config
from pst.training.utils.dim import check_feature_dim
from pst.utils.cli.modes import TrainingMode

logger = logging.getLogger(__name__)


def train_with_cross_validation(config: TrainingMode):
    logger.info("Training with cross validation.")

    trainer_config = init_trainer_config(
        config,
        checkpoint=True,
        early_stopping=True,
        timer=True,
    )

    datamodule = GenomeDataModule(config.data)
    check_feature_dim(config)

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

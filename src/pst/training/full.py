from __future__ import annotations

import logging

from pst.data.modules import GenomeDataModule
from pst.training.utils.dim import check_feature_dim
from pst.training.utils.lightning import init_lightning_trainer
from pst.utils.auto import auto_resolve_model_type
from pst.utils.cli.modes import TrainingMode

logger = logging.getLogger(__name__)


def train_with_all_data(config: TrainingMode):
    if not config.data.train_on_full:
        raise RuntimeError(
            "Cannot train with all data if --train-on-full not passed at command line."
        )

    logger.info("Training a single model with all available data.")

    # shuffling is a must to prevent OOM since the largest data is all at the end
    datamodule = GenomeDataModule(config.data, shuffle=True)

    check_feature_dim(config)

    model_type = auto_resolve_model_type(config.experiment.pst_model_type)
    model = model_type(config.model)

    # update positional embedding max size
    model.check_max_size(datamodule.dataset)

    trainer = init_lightning_trainer(config, checkpoint=True, early_stopping=True)
    trainer.fit(model=model, datamodule=datamodule)

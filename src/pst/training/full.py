import logging
from pathlib import Path

from pst.data.config import DataConfig
from pst.data.modules import GenomeDataModule
from pst.nn.base import BaseModels
from pst.nn.modules import ProteinSetTransformer
from pst.training.utils.lightning import init_lightning_trainer
from pst.training.utils.pst import _add_group_weights
from pst.utils.cli.trainer import TrainerArgs

logger = logging.getLogger(__name__)


def train(
    model: BaseModels,
    data: DataConfig,
    trainer_cfg: TrainerArgs,
):
    datamodule = GenomeDataModule(data, shuffle=True)

    # backwards compatibility with PST that computes group weights
    if isinstance(model, ProteinSetTransformer):
        _add_group_weights(datamodule)

    logger.info(datamodule.summarize())

    log_msg = "Training a single model with "
    if isinstance(data.validation, Path):
        log_msg += f"validation data from {data.validation}."
    elif data.validation == "random":
        log_msg += "a random 80:20 validation split from the input data."
    else:
        log_msg += "all available data WITHOUT validation."

    logger.info(log_msg)

    # update positional embedding max size
    model.check_max_size(datamodule.dataset)

    if datamodule.config.validation is not None:
        checkpoint_monitor = "val_loss"
    else:
        checkpoint_monitor = "train_loss"

    trainer = init_lightning_trainer(
        model.config,
        trainer_cfg,
        checkpoint=True,
        early_stopping=True,
        monitor=checkpoint_monitor,
    )
    trainer.fit(model=model, datamodule=datamodule)

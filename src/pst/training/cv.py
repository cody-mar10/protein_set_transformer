import logging

from lightning_cv import CrossValidationTrainer as CVTrainer

from pst.data.config import CrossValDataConfig
from pst.data.modules import CrossValGenomeDataModule
from pst.nn.base import BaseModelTypes
from pst.nn.config import BaseModelConfig
from pst.nn.modules import ProteinSetTransformer
from pst.training.utils.cv import init_trainer_config
from pst.training.utils.pst import _add_group_weights
from pst.utils.cli.experiment import ExperimentArgs
from pst.utils.cli.trainer import TrainerArgs

logger = logging.getLogger(__name__)


def train_with_cross_validation(
    model_type: BaseModelTypes,
    model_cfg: BaseModelConfig,
    data: CrossValDataConfig,
    trainer_cfg: TrainerArgs,
    experiment: ExperimentArgs,
):
    logger.info("Training with cross validation.")

    trainer_config = init_trainer_config(
        model_cfg,
        experiment,
        trainer_cfg,
        checkpoint=True,
        early_stopping=True,
        timer=True,
    )

    datamodule = CrossValGenomeDataModule(data)

    # backwards compatibility with PST that computes group weights
    if model_type is ProteinSetTransformer:
        _add_group_weights(datamodule)

    trainer = CVTrainer(model_type=model_type, config=trainer_config)

    actual_max_size = datamodule.dataset.max_size
    expected_max_size = model_cfg.max_proteins

    if actual_max_size > expected_max_size:
        logger.warning(
            f"Dataset max size ({actual_max_size}) is greater than model max proteins "
            f"({expected_max_size}). Since a model hasn't been created yet, the model max "
            "proteins will be set to the dataset max size."
        )

        model_cfg.max_proteins = actual_max_size

    trainer.train_with_cross_validation(
        datamodule=datamodule,
        model_config=model_cfg,
    )

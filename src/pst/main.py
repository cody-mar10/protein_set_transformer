import importlib.metadata
import logging
from typing import Generic, TypeVar

from docstring_parser import DocstringStyle
from jsonargparse import CLI, set_docstring_parse_options

from pst import _logger as logger
from pst.data.config import CrossValDataConfig, DataConfig
from pst.nn.base import BaseModels, BaseModelTypes
from pst.nn.config import BaseModelConfig as _SimpleModelConfig
from pst.nn.config import GenomeTripletLossModelConfig as _GenomeTripletLossModelConfig
from pst.nn.config import MaskedLanguageModelingConfig as _MaskedLanguageModelingConfig
from pst.nn.modules import ProteinSetTransformer
from pst.predict.predict import PredictMode
from pst.training import cv, full
from pst.training.finetune import FinetuneMode
from pst.training.tuning import tuning as tuning_module
from pst.utils.cli.experiment import ExperimentArgs
from pst.utils.cli.modes import DownloadMode, EmbedMode, GraphifyMode
from pst.utils.cli.trainer import TrainerArgs
from pst.utils.cli.tuning import TuningArgs
from pst.utils.history import update_config_from_history

# these are needed for dynamic dispatch for modes that require a model config and not a model itself

T_dc = TypeVar("T_dc", bound=_SimpleModelConfig)


class _ModelConfigWrapper(Generic[T_dc]):
    config: T_dc

    def __init__(self, config: T_dc):
        self.config = config


class SimpleModelConfig(_ModelConfigWrapper[_SimpleModelConfig]):
    def __init__(self, config: _SimpleModelConfig):
        super().__init__(config)


class TripletLossModelConfig(_ModelConfigWrapper[_GenomeTripletLossModelConfig]):
    def __init__(self, config: _GenomeTripletLossModelConfig):
        super().__init__(config)


class MaskedLanguageModelingConfig(_ModelConfigWrapper[_MaskedLanguageModelingConfig]):
    def __init__(self, config: _MaskedLanguageModelingConfig):
        super().__init__(config)


class Main(DownloadMode, GraphifyMode, EmbedMode, PredictMode, FinetuneMode):
    """pst CLI to train new PSTs, tune hyperparameters, predict with a pretrained PST, and
    finetune a pretrained PST. Additional utilities to embed protein sequences with ESM2,
    graphify the raw ESM2 protein embeddings for use with PST, and download data and trained
    models from DRYAD"""

    def _check_model_and_config_type(
        self, model_type: BaseModelTypes, model_cfg: _SimpleModelConfig
    ):
        expected_model_config_type = model_type._resolve_model_config_type()
        if not isinstance(model_cfg, expected_model_config_type):
            raise TypeError(
                f"model config must be of type {expected_model_config_type}. Received {type(model_cfg)}"
            )

    def _log_mode(self, mode: str):
        logger.info(f"Running {mode} mode")

    def train(
        self,
        model: BaseModels,
        data: DataConfig,
        trainer: TrainerArgs,
        experiment: ExperimentArgs,
    ):
        """PST simple training mode WITHOUT cross validation"""
        self._log_mode("train")

        if experiment.best_trial is not None:
            # since we are possibly changing the model config, we need to keep track of the dims
            # jsonargparse auto constructs the model, which expands the in/out dims based on the
            # size of the positional embeddings, but we need to undo this expansion
            # we also need to capture the unexpanded dims BEFORE updating the config, since the
            # new config will have the expanded dims with the old config's embed_scale, so the
            # math will be wrong
            model_cfg: _SimpleModelConfig
            model_cfg = model.config
            original_in_dim, original_out_dim = model_cfg._get_original_dims()
            model_cfg, data = update_config_from_history(
                model_config=model_cfg,
                data_config=data,
                history_file=experiment.best_trial,
                tuning_config_def=experiment.config,
            )

            if model_cfg != model.config:
                model_type = type(model)
                # undo dim expansion
                model_cfg.in_dim = original_in_dim
                model_cfg.out_dim = original_out_dim
                model = model_type(model_cfg)

        full.train(model, data, trainer)

    def cv(
        self,
        model: _ModelConfigWrapper,
        data: CrossValDataConfig,
        trainer: TrainerArgs,
        experiment: ExperimentArgs,
        model_type: BaseModelTypes = ProteinSetTransformer,
    ):
        """PST mode for training with cross validation"""
        self._log_mode("cv")
        model_cfg = model.config

        if experiment.best_trial is not None:
            model_cfg, data = update_config_from_history(
                model_config=model_cfg,
                data_config=data,
                history_file=experiment.best_trial,
                tuning_config_def=experiment.config,
            )
            # NOTE: don't need to update config dims because config hasn't
            # been attached to a model yet, so no dim expansion

        self._check_model_and_config_type(model_type=model_type, model_cfg=model_cfg)
        cv.train_with_cross_validation(model_type, model_cfg, data, trainer, experiment)

    def tune(
        self,
        model: _ModelConfigWrapper,
        data: CrossValDataConfig,
        trainer: TrainerArgs,
        experiment: ExperimentArgs,
        tuning: TuningArgs,
        model_type: BaseModelTypes = ProteinSetTransformer,
    ):
        """PST hyperparameter tuning mode with cross validation

        Args:
            model: MODEL
            data: DATA
            trainer: TRAINER
            experiment: EXPERIMENT
            tuning: TUNING
            model_type (BaseModelTypes): PST model type. Defaults to
                `ProteinSetTransformer`.
        """
        self._log_mode("tune")
        model_cfg = model.config
        self._check_model_and_config_type(model_type=model_type, model_cfg=model_cfg)

        tuning_module.tune(model_type, model_cfg, data, trainer, experiment, tuning)


def _setup_logger():
    logger.setLevel(logging.INFO)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] (%(levelname)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)


def main():
    set_docstring_parse_options(style=DocstringStyle.GOOGLE, attribute_docstrings=True)
    _setup_logger()
    CLI(
        Main,
        as_positional=False,
        version=importlib.metadata.version("ptn-set-transformer"),
    )


if __name__ == "__main__":
    main()

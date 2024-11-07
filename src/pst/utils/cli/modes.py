from __future__ import annotations

import pydantic_argparse

from pst.data.modules import DataConfig
from pst.embed import EmbedArgs
from pst.nn.config import (
    GenomeTripletLossConfig,
    MaskedLanguageModelingConfig,
    MaskedLanguageModelingLossConfig,
    ModelConfig,
)
from pst.utils.cli.download import DownloadArgs
from pst.utils.cli.experiment import ExperimentArgs
from pst.utils.cli.finetune import FinetuningArgs
from pst.utils.cli.graphify import GraphifyArgs
from pst.utils.cli.predict import PredictArgs
from pst.utils.cli.trainer import TrainerArgs
from pst.utils.cli.tuning import TuningArgs


# take all args for all loss configs so they are available at cli
class _UnionLossConfig(GenomeTripletLossConfig, MaskedLanguageModelingLossConfig):
    pass


# take all args for all model configs so they are available at cli
# NOTE: cannot have fields of same name refer to different things
class UnionModelConfig(ModelConfig, MaskedLanguageModelingConfig):
    loss: _UnionLossConfig


class TrainingMode(pydantic_argparse.BaseCommand):
    model: UnionModelConfig
    data: DataConfig
    trainer: TrainerArgs
    experiment: ExperimentArgs


class TuningMode(TrainingMode):
    tuning: TuningArgs


class InferenceMode(TrainingMode):
    predict: PredictArgs


class FinetuningMode(TrainingMode):
    finetuning: FinetuningArgs


class PreprocessingMode(pydantic_argparse.BaseCommand):
    graphify: GraphifyArgs


class DownloadMode(pydantic_argparse.BaseCommand):
    download: DownloadArgs


class EmbedMode(pydantic_argparse.BaseCommand):

    embed: EmbedArgs

from __future__ import annotations

import pydantic_argparse

from pst.data.modules import DataConfig
from pst.nn.config import ModelConfig
from pst.utils.cli.experiment import ExperimentArgs
from pst.utils.cli.graphify import GraphifyArgs
from pst.utils.cli.predict import PredictArgs
from pst.utils.cli.trainer import TrainerArgs
from pst.utils.cli.tuning import TuningArgs


class TrainingMode(pydantic_argparse.BaseCommand):
    model: ModelConfig
    data: DataConfig
    trainer: TrainerArgs
    experiment: ExperimentArgs


class TuningMode(TrainingMode):
    tuning: TuningArgs


class InferenceMode(TrainingMode):
    predict: PredictArgs


class PreprocessingMode(pydantic_argparse.BaseCommand):
    graphify: GraphifyArgs

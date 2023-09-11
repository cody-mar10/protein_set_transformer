from __future__ import annotations

import pydantic_argparse

from pst.arch.data import DataConfig
from pst.arch.modules import ModelConfig

from .experiment import ExperimentArgs
from .predict import PredictArgs
from .trainer import TrainerArgs
from .tuning import TuningArgs


class TrainingMode(pydantic_argparse.BaseCommand):
    model: ModelConfig
    data: DataConfig
    trainer: TrainerArgs
    experiment: ExperimentArgs


class TuningMode(TrainingMode):
    tuning: TuningArgs


class InferenceMode(TrainingMode):
    predict: PredictArgs

from __future__ import annotations

from typing import Optional

import pydantic_argparse
from pydantic import BaseModel, Field

from .augmentation import AugmentationArgs
from .data import DataArgs
from .experiment import ExperimentArgs
from .loss import LossArgs
from .model import ModelArgs
from .optimizer import OptimizerArgs
from .predict import PredictArgs
from .trainer import TrainerArgs
from .utils import get_models, split_fields


class TrainCommand(
    ModelArgs,
    LossArgs,
    OptimizerArgs,
    AugmentationArgs,
    DataArgs,
    TrainerArgs,
    ExperimentArgs,
):
    pass


class PredictCommand(TrainCommand, PredictArgs):
    pass


class TrainingMode(BaseModel):
    model: ModelArgs
    loss: LossArgs
    optimizer: OptimizerArgs
    augmentation: AugmentationArgs
    data: DataArgs
    trainer: TrainerArgs
    experiment: ExperimentArgs


class InferenceMode(TrainingMode):
    predict: PredictArgs


class Args(BaseModel):
    train: Optional[TrainCommand] = Field(description="train mode")
    predict: Optional[PredictCommand] = Field(description="predict/inference mode")


def parse_args() -> Args:
    parser = pydantic_argparse.ArgumentParser(
        model=Args,
        description=(
            "Train or predict genome-level embeddings based on sets of protein-level "
            "embeddings"
        ),
    )

    args = parser.parse_typed_args()
    return args


def convert(args: Args):
    model_types = get_models()
    if args.train is not None:
        config = split_fields(args.train)
        convert_cls = TrainingMode

    else:
        config = split_fields(args.predict)
        convert_cls = InferenceMode

    models = {name: model_types[name](**kwargs) for name, kwargs in config.items()}
    return convert_cls(**models)

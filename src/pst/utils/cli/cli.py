from __future__ import annotations

from typing import Optional

import pydantic_argparse
from pydantic import BaseModel, Field

from pst.arch.data import DataConfig
from pst.arch.modules import ModelConfig

from .experiment import ExperimentArgs
from .predict import PredictArgs
from .trainer import TrainerArgs


class TrainingMode(pydantic_argparse.BaseCommand):
    model: ModelConfig
    data: DataConfig
    trainer: TrainerArgs
    experiment: ExperimentArgs


class InferenceMode(TrainingMode):
    predict: PredictArgs


class Args(BaseModel):
    train: Optional[TrainingMode] = Field(None, description="train mode")
    predict: Optional[InferenceMode] = Field(None, description="predict/inference mode")


def parse_args(args: Optional[list[str]] = None) -> Args:
    parser = pydantic_argparse.ArgumentParser(
        model=Args,
        description=(
            "Train or predict genome-level embeddings based on sets of protein-level "
            "embeddings"
        ),
    )

    return parser.parse_typed_args(args)

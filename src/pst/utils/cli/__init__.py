from __future__ import annotations

from typing import Optional

import pydantic_argparse
from pydantic import BaseModel, Field

from pst.utils.cli.modes import (
    InferenceMode,
    PreprocessingMode,
    TrainingMode,
    TuningMode,
)


class Args(BaseModel):
    train: Optional[TrainingMode] = Field(None, description="train mode")
    tune: Optional[TuningMode] = Field(None, description="tune mode")
    predict: Optional[InferenceMode] = Field(None, description="predict/inference mode")
    graphify: Optional[PreprocessingMode] = Field(
        None,
        description="Pre-processing mode to convert raw ESM2 protein embeddings into a graph-formatted dataset to be used as input for the other modes",
    )


def parse_args(args: Optional[list[str]] = None) -> Args:
    parser = pydantic_argparse.ArgumentParser(
        model=Args,
        description=(
            "Train or predict genome-level embeddings based on sets of protein-level "
            "embeddings"
        ),
    )

    return parser.parse_typed_args(args)

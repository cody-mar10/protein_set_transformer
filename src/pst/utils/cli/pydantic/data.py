from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .utils import register_model

EdgeIndexStrategy = Literal["chunked", "sparse", "full"]

_NAME = "data"


@register_model(_NAME)
class DataArgs(BaseModel):
    file: Path = Field(
        ...,
        description="input protein embeddings file in .h5 file format with the fields .data for protein embeddings",  # TODO: add more desc  # noqa: E501
    )
    batch_size: int = Field(
        32, description="batch size in number of genomes", gt=4, le=128, multiple_of=2
    )
    train_on_full: bool = Field(
        False,
        description=(
            "whether to train a single model on the full input data. When not "
            "specified, the default is  to train multiple models with cross "
            "validation"
        ),
    )
    pin_memory: bool = Field(
        True, description="whether to pin memory onto a CUDA GPU or not"
    )
    num_workers: int = Field(0, description="additional cpu workers to load data", ge=0)
    edge_strategy: EdgeIndexStrategy = Field(
        "chunked",
        description=(
            "strategy to create 'edges' between protein nodes in a genome graph. "
            "chunked = split genomes in --chunk-size chunks. sparse = remove "
            "interactions longer than --threshold. full = fully connected graph "
            "like a regular transformer."
        ),
    )
    chunk_size: int = Field(
        30,
        description="size of sub-chunks to break genomes into if using --edge_strategy chunked",  # noqa: E501
        ge=15,
        le=50,
    )
    threshold: int = Field(
        -1,
        description="range of protein interactions if using --edge_strategy [chunked|sparse]",  # noqa: E501,
    )
    log_inverse: bool = Field(
        False, description="take the log of inverse class freqs as weights"
    )

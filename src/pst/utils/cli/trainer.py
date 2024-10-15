from __future__ import annotations

from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

AcceleratorOpts = Literal["cpu", "gpu", "tpu", "auto"]
PrecisionOpts = Literal["16-mixed", "bf16-mixed", "32"]
StrategyOpts = Literal["ddp", "ddp_spawn", "ddp_notebook", "fsdp", "auto"]
GradClipAlgOpts = Literal["norm", "value"]


class MaxTimeOpts(Enum):
    short = timedelta(hours=12)
    medium = timedelta(days=1)
    long = timedelta(weeks=1)
    none = None

    def __repr__(self) -> str:
        return self.name


class TrainerArgs(BaseModel):
    devices: int = Field(
        1,
        description="number of accelerator devices to use. For CPUs, this sets the total thread usage.",  # noqa: E501
    )
    accelerator: AcceleratorOpts = Field("auto", description="accelerator to use")
    default_root_dir: Path = Field(
        Path("lightning_root"),
        description="lightning root dir for checkpointing and logging",
    )
    max_epochs: int = Field(1000, description="max number of training epochs", gt=0)
    precision: PrecisionOpts = Field("16-mixed", description="floating point precision")
    strategy: StrategyOpts = Field("ddp", description="parallelized training strategy")
    gradient_clip_algorithm: Optional[GradClipAlgOpts] = Field(
        "norm", description="optional gradient clipping procedure"
    )
    gradient_clip_val: Optional[float] = Field(
        0.5, description="optional value if clipping gradients"
    )
    max_time: MaxTimeOpts = Field(
        MaxTimeOpts.none,
        description=(
            "maximum allowed training time"
            "[choices: short=12h, medium=1d, long=7d, none=no limit]"
        ),
    )
    limit_train_batches: Optional[int | float] = Field(
        None,
        description=(
            "optional limit to number of training batches. An integer means "
            "train with that number of training batches, while a float between "
            "(0.0, 1.0] means that fraction of the training data is used)"
        ),
    )
    limit_val_batches: Optional[int | float] = Field(
        None,
        description=(
            "optional limit to number of val batches. An integer means "
            "validate with that number of val batches, while a float between "
            "(0.0, 1.0] means that fraction of the val data is used)"
        ),
    )
    accumulate_grad_batches: int = Field(
        1,
        description="number of batches to accumulate gradients before the optimizer steps",
    )

    @field_validator("max_time", mode="before")
    def _convert(cls, value: str | MaxTimeOpts) -> MaxTimeOpts:
        if isinstance(value, str):
            return MaxTimeOpts[value]
        return value

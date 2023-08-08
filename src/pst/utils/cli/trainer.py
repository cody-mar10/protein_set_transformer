from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator

AcceleratorOpts = Literal["cpu", "gpu", "tpu", "auto"]
PrecisionOpts = Literal["16-mixed", "bf16-mixed", "32"]
StrategyOpts = Literal["ddp", "ddp_spawn", "ddp_notebook", "fsdp", "auto"]
GradClipAlgOpts = Literal["norm", "value"]
MaxTimeOpts = Literal["short", "medium", "long", None]


class TrainerArgs(BaseModel):
    devices: int = Field(
        1,
        description="number of accelerator devices to use. For CPUs, this sets the total thread usage.",  # noqa: E501
    )
    accelerator: AcceleratorOpts = Field("gpu", description="accelerator to use")
    default_root_dir: Path = Field(
        Path("lightning_root"),
        description="lightning root dir for checkpointing and logging",
    )
    max_epochs: int = Field(1000, description="max number of training epochs", gt=0)
    precision: PrecisionOpts = Field("16-mixed", description="floating point precision")
    strategy: StrategyOpts = Field("ddp", description="parallelized training strategy")
    gradient_clip_algorithm: Optional[GradClipAlgOpts] = Field(
        None, description="optional gradient clipping procedure"
    )
    gradient_clip_val: Optional[float] = Field(
        None, description="optional value if clipping gradients"
    )
    max_time: Optional[timedelta] = Field(
        None,
        description=(
            "maximum allowed training time"
            "[choices: short=12h, medium=1d, long=7d, None=no limit]"
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

    @validator("max_time", pre=True)
    def convert_max_time(cls, value: MaxTimeOpts) -> Optional[timedelta]:
        # CHTC guidelines
        # add a small buffer to allow clean up
        buffer = timedelta(minutes=15)

        match value:
            case "short":
                limit = timedelta(hours=12) - buffer
                return limit
            case "medium":
                limit = timedelta(days=1) - buffer
                return limit
            case "long":
                limit = timedelta(weeks=1) - buffer
                return limit
            case None:
                return None
            case _:
                raise ValueError(
                    f"--max-time must be one of the following: {MaxTimeOpts}"
                )

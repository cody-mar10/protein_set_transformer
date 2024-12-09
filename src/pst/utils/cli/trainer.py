from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import torch
from jsonargparse.typing import (
    NonNegativeInt,
    OpenUnitInterval,
    PositiveFloat,
    PositiveInt,
)

from pst.utils.dataclass_utils import DataclassValidatorMixin, validated_field

GradClipAlgOpts = Literal["norm", "value"]


class AcceleratorOpts(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"
    auto = "auto"


class PrecisionOpts(str, Enum):
    bf16 = "bf16-mixed"
    fp16 = "16-mixed"
    fp32 = "32"


class StrategyOpts(str, Enum):
    ddp = "ddp"
    ddp_spawn = "ddp_spawn"
    ddp_notebook = "ddp_notebook"
    fsdp = "fsdp"
    auto = "auto"


class MaxTimeOpts(Enum):
    short = timedelta(hours=12)
    medium = timedelta(days=1)
    long = timedelta(weeks=1)
    none = None

    def __repr__(self) -> str:
        return self.name


class SWAAnnealingOpts(str, Enum):
    linear = "linear"
    cos = "cos"


@dataclass
class ExtraTrainerArgs(DataclassValidatorMixin):
    """EXTRA TRAINER ARGS"""

    # most of this is being moved from Experiment since they really apply to trainers

    patience: int = validated_field(5, validator=PositiveInt)
    """early stopping patience, see lightning description"""

    save_top_k: int = validated_field(1, validator=NonNegativeInt)
    """save the best k models"""

    swa: bool = False
    """use stochastic weight averaging"""

    swa_epoch_start: float = validated_field(0.8, validator=OpenUnitInterval)
    """when to start SWA if enabled. Integers are an epoch number, while floats in range (0.0, 1.0) are a fraction of max epochs."""

    annealing_epochs: int = validated_field(10, validator=PositiveInt)
    """number of epochs to anneal learning rate during swa if enabled"""

    annealing_strategy: SWAAnnealingOpts = SWAAnnealingOpts.linear
    """annealing strategy using during swa if enabled"""

    min_delta: float = validated_field(0.05, validator=PositiveFloat)
    """minimum change in monitored quantity to qualify as an improvement. There must be an
    improvement in at least --patience epochs before an EarlyStopping callback is checked
    """


@dataclass
class TrainerArgs(DataclassValidatorMixin):
    """TRAINER"""

    devices: int = validated_field(1, validator=PositiveInt)
    """number of accelerator devices to use. For CPUs, this sets the total thread usage."""

    accelerator: AcceleratorOpts = AcceleratorOpts.auto
    """accelerator to use"""

    default_root_dir: Path = Path("lightning_root")
    """lightning root dir for checkpointing and logging"""

    max_epochs: int = validated_field(1000, validator=PositiveInt)
    """max number of training epochs"""

    precision: PrecisionOpts = PrecisionOpts.bf16
    """floating point precision"""

    strategy: StrategyOpts = StrategyOpts.auto
    """parallelized training strategy"""

    gradient_clip_algorithm: Optional[GradClipAlgOpts] = "norm"
    """optional gradient clipping procedure"""

    gradient_clip_val: Optional[float] = 0.5
    """optional value if clipping gradients"""

    max_time: MaxTimeOpts = MaxTimeOpts.none
    """maximum allowed training time"""

    limit_train_batches: Optional[int | float] = None
    """optional limit to number of training batches. An integer means train with that number of
    training batches, while a float between (0.0, 1.0] means that fraction of the training data
    is used"""

    limit_val_batches: Optional[int | float] = None
    """optional limit to number of val batches. An integer means validate with that number of val
    batches, while a float between (0.0, 1.0] means that fraction of the val data is used"""

    accumulate_grad_batches: int = validated_field(1, validator=PositiveInt)
    """number of batches to accumulate gradients over"""

    extra: ExtraTrainerArgs = field(default_factory=ExtraTrainerArgs)
    """EXTRA TRAINER ARGS"""

    def _validate_limit_batches(
        self, attr: Literal["limit_train_batches", "limit_val_batches"]
    ):
        value = getattr(self, attr)
        if value is not None:
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{attr} must be >= 0 if an integer")
            elif isinstance(value, float) and (value <= 0 or value > 1):
                raise ValueError(f"{attr} must be in (0.0, 1.0] if a float")

    def _validate_accelerator(self):
        if self.accelerator == "auto":
            self.accelerator = (
                AcceleratorOpts.gpu
                if torch.cuda.is_available()
                else AcceleratorOpts.cpu
            )

    def _check_cpu_accelerator(self):
        if self.accelerator == "cpu":
            threads = self.devices
            self.precision = PrecisionOpts.fp32
            torch.set_num_threads(threads)
            self.devices = 1
            self.strategy = StrategyOpts.auto

    def __post_init__(self):
        super().__post_init__()

        for attr in ["limit_train_batches", "limit_val_batches"]:
            self._validate_limit_batches(attr)  # type: ignore

        if self.gradient_clip_val is not None and self.gradient_clip_val < 0:
            raise ValueError("gradient_clip_val must be >= 0 if specified")

        self._validate_accelerator()
        self._check_cpu_accelerator()

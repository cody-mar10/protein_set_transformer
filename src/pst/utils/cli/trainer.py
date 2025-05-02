from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from attrs import define, field, validators

from pst.utils.attrs.converters import EnumConverter, enum_field
from pst.utils.attrs.dataclass_utils import AttrsDataclassUtilitiesMixin
from pst.utils.attrs.validators import (
    non_negative_int,
    open_unit_interval,
    positive_float,
    positive_int,
)

GradClipAlgOpts = Literal["norm", "value"]


class AcceleratorOpts(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"
    auto = "auto"

    @staticmethod
    def convert_auto(instance: Union["AcceleratorOpts", str]) -> "AcceleratorOpts":
        instance = EnumConverter.convert(instance, AcceleratorOpts)

        if instance == AcceleratorOpts.auto:
            return AcceleratorOpts.gpu if torch.cuda.is_available() else AcceleratorOpts.cpu
        return instance


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


@define
class ExtraTrainerArgs(AttrsDataclassUtilitiesMixin):
    """EXTRA TRAINER ARGS"""

    # most of this is being moved from Experiment since they really apply to trainers

    patience: int = field(default=5, validator=positive_int)
    """early stopping patience, see lightning description"""

    save_top_k: int = field(default=1, validator=non_negative_int)
    """save the best k models"""

    swa: bool = False
    """use stochastic weight averaging"""

    swa_epoch_start: float = field(default=0.8, validator=open_unit_interval)
    """when to start SWA if enabled. Integers are an epoch number, while floats in range (0.0, 1.0) are a fraction of max epochs."""

    annealing_epochs: int = field(default=10, validator=positive_int)
    """number of epochs to anneal learning rate during swa if enabled"""

    annealing_strategy: SWAAnnealingOpts = enum_field(
        enum_cls=SWAAnnealingOpts, default=SWAAnnealingOpts.linear
    )
    """annealing strategy using during swa if enabled"""

    min_delta: float = field(default=0.05, validator=positive_float)
    """minimum change in monitored quantity to qualify as an improvement. There must be an
    improvement in at least --patience epochs before an EarlyStopping callback is checked
    """


_left_open_right_closed_unit_interval_validator = validators.and_(
    validators.instance_of(float),
    validators.gt(0.0),
    validators.le(1.0),
)
_limit_batches_validator = validators.optional(
    validators.or_(
        non_negative_int,
        _left_open_right_closed_unit_interval_validator,  # type: ignore
    )
)


@define
class TrainerArgs(AttrsDataclassUtilitiesMixin):
    """TRAINER"""

    devices: int = field(default=1, validator=positive_int)
    """number of accelerator devices to use. For CPUs, this sets the total thread usage."""

    accelerator: AcceleratorOpts = field(
        default=AcceleratorOpts.auto, converter=AcceleratorOpts.convert_auto
    )
    """accelerator to use"""

    default_root_dir: Path = Path("lightning_root")
    """lightning root dir for checkpointing and logging"""

    max_epochs: int = field(default=1000, validator=positive_int)
    """max number of training epochs"""

    precision: PrecisionOpts = enum_field(enum_cls=PrecisionOpts, default=PrecisionOpts.bf16)
    """floating point precision"""

    strategy: StrategyOpts = enum_field(enum_cls=StrategyOpts, default=StrategyOpts.auto)
    """parallelized training strategy"""

    gradient_clip_algorithm: Optional[GradClipAlgOpts] = "norm"
    """optional gradient clipping procedure"""

    gradient_clip_val: Optional[float] = field(
        default=0.5, validator=validators.optional(positive_float)
    )
    """optional value if clipping gradients"""

    max_time: MaxTimeOpts = enum_field(enum_cls=MaxTimeOpts, default=MaxTimeOpts.none)
    """maximum allowed training time"""

    limit_train_batches: Union[int, float, None] = field(
        default=None, validator=_limit_batches_validator
    )
    """optional limit to number of training batches. An integer means train with that number of
    training batches, while a float between (0.0, 1.0] means that fraction of the training data
    is used"""

    limit_val_batches: Union[int, float, None] = field(
        default=None, validator=_limit_batches_validator
    )
    """optional limit to number of val batches. An integer means validate with that number of val
    batches, while a float between (0.0, 1.0] means that fraction of the val data is used"""

    accumulate_grad_batches: int = field(default=1, validator=positive_int)
    """number of batches to accumulate gradients over"""

    extra: ExtraTrainerArgs = field(factory=ExtraTrainerArgs)
    """EXTRA TRAINER ARGS"""

    def _check_cpu_accelerator(self):
        if self.accelerator == AcceleratorOpts.cpu:
            threads = self.devices
            self.precision = PrecisionOpts.fp32
            torch.set_num_threads(threads)
            self.devices = 1
            self.strategy = StrategyOpts.auto
        else:
            torch.set_num_threads(1)

    def __attrs_post_init__(self):
        self._check_cpu_accelerator()

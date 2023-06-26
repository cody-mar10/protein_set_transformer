from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

from .utils import asdict, register_defaults, register_handler

AcceleratorOpts = Literal["cpu", "gpu", "tpu", "auto"]
PrecisionOpts = Literal["16-mixed", "bf16-mixed", 32, "32"]
StrategyOpts = Literal["ddp", "ddp_spawn", "ddp_notebook", "fsdp"]
GradClipAlgOpts = Literal["norm", "value"]
MaxTimeOpts = Literal["short", "medium", "long", None]


# these are directly passed to the lightning.Trainer class as kwargs
@dataclass
class TrainerArgs:
    devices: int = 1
    accelerator: AcceleratorOpts = "gpu"
    default_root_dir: Path = Path("lightning_root")
    max_epochs: int = 1000
    precision: PrecisionOpts = "16-mixed"
    strategy: StrategyOpts = "ddp"
    gradient_clip_algorithm: Optional[GradClipAlgOpts] = None
    gradient_clip_val: Optional[float] = None
    max_time: Optional[timedelta] = None


_DEFAULTS = TrainerArgs()
register_defaults(_DEFAULTS, "trainer")


def _convert_max_time(
    val: MaxTimeOpts,
    buffer_amount: int = 15,
    buffer_type: Literal["days", "hours", "minutes"] = "minutes",
) -> Optional[timedelta]:
    # these are based on CHTC guidelines

    # add a small buffer to stop training and save outputs in chtc
    buffer_kwargs = {buffer_type: buffer_amount}
    buffer = timedelta(**buffer_kwargs)
    match val:
        case "short":
            limit = timedelta(hours=12) - buffer
            return limit
        case "medium":
            limit = timedelta(days=1) - buffer
            return limit
        case "long":
            limit = timedelta(weeks=1) - buffer
            return limit
        case _:
            return None


@register_handler
def add_trainer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("TRAINER ARGS")
    group.add_argument(
        "--devices",
        metavar="INT",
        type=int,
        default=_DEFAULTS.devices,
        help=(
            "number of accelerator devices to use. For CPUs, this sets the total thread"
            " usage. (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--accelerator",
        metavar="DEVICE",
        choices={"cpu", "gpu", "tpu", "auto"},
        default=_DEFAULTS.accelerator,
        help="accelerator to use (default: %(default)s) [choices: %(choices)s]",
    )
    group.add_argument(
        "--default_root_dir",
        metavar="DIR",
        type=Path,
        default=_DEFAULTS.default_root_dir,
        help="lightning root dir for model checkpointing (default: %(default)s)",
    )
    group.add_argument(
        "--max_epochs",
        metavar="INT",
        type=int,
        default=_DEFAULTS.max_epochs,
        help="max number of training epochs (default: %(default)s)",
    )
    group.add_argument(
        "--strategy",
        metavar="",
        choices={"ddp", "ddp_spawn", "ddp_notebook", "fsdp"},
        default=_DEFAULTS.strategy,
        help=(
            "parallelized training strategy (default: %(default)s) "
            "[choices: %(choices)s]"
        ),
    )
    group.add_argument(
        "--precision",
        metavar="",
        choices={"16-mixed", 32, "bf16-mixed"},
        default=_DEFAULTS.precision,
        type=lambda x: int(x) if x.isdigit() else x,
        help="floating point precision (default: %(default)s) [choices: %(choices)s]",
    )
    group.add_argument(
        "--gradient-clip-algorithm",
        metavar="",
        choices={"norm", "value"},
        help=(
            "optional procedure to clip gradients during (default: %(default)s) "
            "[choices: %(choices)s]"
        ),
    )
    group.add_argument(
        "--gradient-clip-val",
        metavar="FLOAT",
        type=float,
        help="optional value used if clipping gradients (default: %(default)s)",
    )
    group.add_argument(
        "--max-time",
        metavar="",
        choices={"short", "medium", "long", None, "None"},
        default=_DEFAULTS.max_time,
        help=(
            "maximum amount of time for training (default: %(default)s) "
            "[choices: short=12h, medium=1d, long=7d, None=no limit]"
        ),
    )


@asdict
def parse_trainer_args(args: argparse.Namespace) -> TrainerArgs:
    targs = TrainerArgs(
        devices=args.devices,
        accelerator=args.accelerator,
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        precision=args.precision,
        strategy=args.strategy,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        gradient_clip_val=args.gradient_clip_val,
        max_time=_convert_max_time(args.max_time),
    )

    return targs

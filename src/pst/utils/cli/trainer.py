from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .utils import asdict, register


# these are directly passed to the lightning.Trainer class as kwargs
@dataclass
class TrainerArgs:
    devices: int = 1
    accelerator: Literal["cpu", "gpu", "tpu", "auto"] = "gpu"
    default_root_dir: Path = Path("lightning_root")
    max_epochs: int = 1000
    precision: Literal["16-mixed", "bf16-mixed", 32, "32"] = "16-mixed"
    strategy: Literal["ddp", "ddp_spawn"] = "ddp"
    gradient_clip_algorithm: Optional[Literal["norm", "value"]] = None
    gradient_clip_val: Optional[float] = None


_DEFAULTS = TrainerArgs()


@register
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
        choices={"ddp", "ddp_spawn"},
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


@asdict
def parse_trainer_args(args: argparse.Namespace) -> TrainerArgs:
    return TrainerArgs(
        devices=args.devices,
        accelerator=args.accelerator,
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        precision=args.precision,
        strategy=args.strategy,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        gradient_clip_val=args.gradient_clip_val,
    )

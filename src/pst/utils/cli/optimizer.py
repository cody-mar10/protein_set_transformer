from __future__ import annotations

import argparse
from dataclasses import dataclass

from .utils import asdict, register


@dataclass
class OptimizerArgs:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 0


_DEFAULTS = OptimizerArgs()


@register
def add_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("OPTIMIZER ARGS")
    group.add_argument(
        "--lr",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.lr,
        help="learning rate (default: %(default)s)",
    )
    group.add_argument(
        "--weight_decay",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.weight_decay,
        help="optimizer weight decay (default: %(default)s)",
    )
    group.add_argument(
        "--warmup_steps",
        metavar="INT",
        type=int,
        default=_DEFAULTS.warmup_steps,
        help="number of training steps to warm-up the learning rate to the max (default: %(default)s)",
    )
    group.add_argument(
        "--betas",
        metavar="[FLOAT FLOAT]",
        type=float,
        nargs=2,
        default=_DEFAULTS.betas,
        help="optimizer betas (default: %(default)s)",
    )


@asdict
def parse_optimizer_args(args: argparse.Namespace) -> OptimizerArgs:
    return OptimizerArgs(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        warmup_steps=args.warmup_steps,
    )

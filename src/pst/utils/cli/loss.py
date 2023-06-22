from __future__ import annotations

import argparse
from dataclasses import dataclass

from utils import asdict, register


@dataclass
class LossArgs:
    margin: float = 0.1


_DEFAULTS = LossArgs()


@register
def add_loss_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("LOSS ARGS")
    group.add_argument(
        "--margin",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.margin,
        help="triplet loss margin (default: %(default)s)",
    )


@asdict
def parse_loss_args(args: argparse.Namespace) -> LossArgs:
    return LossArgs(margin=args.margin)

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from .utils import asdict, register_defaults, register_handler

NO_NEGATIVES_MODES = Literal["closest_to_positive", "closest_to_anchor"]


@dataclass
class AugmentationArgs:
    sample_scale: float = 7.0
    sample_rate: float = 0.5
    no_negatives_mode: NO_NEGATIVES_MODES = "closest_to_positive"


_DEFAULTS = AugmentationArgs()
register_defaults(_DEFAULTS, "augmentation")


@register_handler
def add_augmentation_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("AUGMENTATION ARGS")
    group.add_argument(
        "--sample_scale",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.sample_scale,
        help=(
            "exponential decay scale factor for weighting negative samples during loss "
            "(default: %(default)s)"
        ),
    )
    group.add_argument(
        "--sample_rate",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.sample_rate,
        help="PointSwap sampler swapping rate (default: %(default)s)",
    )
    group.add_argument(
        "--no-negatives-mode",
        metavar="",
        choices={"closest_to_positive", "closest_to_anchor"},
        default=_DEFAULTS.no_negatives_mode,
        help=(
            "mode to handle event of no semihard negative sample existing "
            "(default: %(default)s)"
        ),
    )


@asdict
def parse_augmentation_args(args: argparse.Namespace) -> AugmentationArgs:
    return AugmentationArgs(
        sample_scale=args.sample_scale,
        sample_rate=args.sample_rate,
        no_negatives_mode=args.no_negatives_mode,
    )

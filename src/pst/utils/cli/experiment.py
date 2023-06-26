from __future__ import annotations

import argparse
from dataclasses import dataclass

from .utils import asdict, register_defaults, register_handler


@dataclass
class ExperimentArgs:
    name: str = "exp0"
    patience: int = 5


_DEFAULTS = ExperimentArgs()
register_defaults(_DEFAULTS, "experiment")


@register_handler
def add_experiment_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("EXPERIMENT ARGS")
    group.add_argument(
        "--exp-name",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.name,
        help="experiment name in logging directory during training (default: %(default)s)",  # noqa: E501
    )
    group.add_argument(
        "--patience",
        metavar="INT",
        type=int,
        default=_DEFAULTS.patience,
        help="early stopping patience, see lightning description (default: %(default)s)",  # noqa: E501
    )


@asdict
def parse_experiment_args(args: argparse.Namespace) -> ExperimentArgs:
    return ExperimentArgs(name=args.exp_name, patience=args.patience)

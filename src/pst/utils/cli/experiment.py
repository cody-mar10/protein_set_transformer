from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from .utils import asdict, register_defaults, register_handler

AnnealingOpts = Literal["linear", "cos"]


@dataclass
class ExperimentArgs:
    name: str = "exp0"
    patience: int = 5
    save_top_k: int = 3
    swa: bool = False
    swa_epoch_start: int | float = 0.8
    annealing_epochs: int = 10
    annealing_strategy: AnnealingOpts = "linear"


_DEFAULTS = ExperimentArgs()
register_defaults(_DEFAULTS, "experiment")


@register_handler
def add_experiment_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("EXPERIMENT ARGS")
    group.add_argument(
        "--exp-name",
        metavar="NAME",
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
    group.add_argument(
        "--save-top-k",
        metavar="INT",
        type=int,
        default=_DEFAULTS.save_top_k,
        help="save the best k models (default: %(default)s)",
    )

    ### SWA args
    group.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging (default: %(default)s)",
    )
    group.add_argument(
        "--swa-epoch-start",
        metavar="INT|FLOAT",
        type=lambda x: int(x) if x.isdigit() else float(x),
        default=_DEFAULTS.swa_epoch_start,
        help=(
            "when to start SWA if enabled. Integers are an epoch number, "
            "while floats in range (0.0, 1.0) are a fraction of max epochs. "
            "(default: %(default)s)"
        ),
    )
    group.add_argument(
        "--annealing-epochs",
        metavar="INT",
        type=int,
        default=_DEFAULTS.annealing_epochs,
        help=(
            "number of epochs to anneal learning rate during swa if enabled "
            "(default: %(default)s)"
        ),
    )
    group.add_argument(
        "--annealing-strategy",
        metavar="",
        default=_DEFAULTS.annealing_strategy,
        choices={"linear", "cos"},
        help=(
            "annealing strategy using during swa if enabled "
            "(default: %(default)s) [choices: %(choices)s]"
        ),
    )


@asdict
def parse_experiment_args(args: argparse.Namespace) -> ExperimentArgs:
    return ExperimentArgs(
        name=args.exp_name,
        patience=args.patience,
        save_top_k=args.save_top_k,
        swa=args.swa,
        swa_epoch_start=args.swa_epoch_start,
        annealing_epochs=args.annealing_epochs,
        annealing_strategy=args.annealing_strategy,
    )

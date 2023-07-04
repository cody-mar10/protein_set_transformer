from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, Optional, get_args

from .utils import (
    asdict,
    parse_int_or_float_arg,
    register_defaults,
    register_handler,
    validate_proportion_range,
)

AnnealingOpts = Literal["linear", "cos"]


@dataclass
class ExperimentArgs:
    config: Optional[Path] = None
    name: str = "exp0"
    patience: int = 5
    save_top_k: int = 3
    swa: bool = False
    swa_epoch_start: int | float = 0.8
    annealing_epochs: int = 10
    annealing_strategy: AnnealingOpts = "linear"
    tune: bool = True
    n_trials: int = 100
    prune: bool = True
    parallel: bool = False


_DEFAULTS = ExperimentArgs()
register_defaults(_DEFAULTS, "experiment")


@register_handler
def add_experiment_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("EXPERIMENT ARGS")
    group.add_argument(
        "--config",
        metavar="FILE",
        help="optional config toml file to specify which hyperparameters to tune with optuna",  # noqa: E501
    )
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

    # TUNING ARGS
    group.add_argument(
        "--no-tune",
        action="store_true",
        help="don't use hyperparameter sampling/tuning during training (default: %(default)s)",  # noqa: E501
    )
    group.add_argument(
        "--n-trials",
        metavar="INT",
        type=int,
        default=_DEFAULTS.n_trials,
        help="number of tuning trials is tuning (default: %(default)s)",
    )
    group.add_argument(
        "--no-prune",
        action="store_true",
        help="don't use early pruning during tuning for bad hyperparameter sets (default: %(default)s)",  # noqa: E501
    )
    group.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "if multiple GPUs are available, run tuning trials in parallel "
            "determined by the number of GPUs requested (default: %(default)s)"
        ),
    )

    ### SWA args
    group.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging (default: %(default)s)",
    )

    range_check = partial(
        validate_proportion_range, left_inclusive=False, right_inclusive=False
    )
    parse_int_or_float_arg_with_range_check = partial(
        parse_int_or_float_arg, float_predicate=range_check
    )
    group.add_argument(
        "--swa-epoch-start",
        metavar="INT|FLOAT",
        type=parse_int_or_float_arg_with_range_check,
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
        choices=get_args(AnnealingOpts),
        help=(
            "annealing strategy using during swa if enabled "
            "(default: %(default)s) [choices: %(choices)s]"
        ),
    )


@asdict
def parse_experiment_args(args: argparse.Namespace) -> ExperimentArgs:
    return ExperimentArgs(
        config=args.config,
        name=args.exp_name,
        patience=args.patience,
        save_top_k=args.save_top_k,
        tune=not args.no_tune,
        n_trials=args.n_trials,
        prune=not args.no_prune,
        parallel=args.parallel,
        swa=args.swa,
        swa_epoch_start=args.swa_epoch_start,
        annealing_epochs=args.annealing_epochs,
        annealing_strategy=args.annealing_strategy,
    )

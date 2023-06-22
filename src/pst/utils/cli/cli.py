from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal

from augmentation import parse_augmentation_args
from data import parse_data_args
from experiment import parse_experiment_args
from loss import parse_loss_args
from model import parse_model_args
from optimizer import parse_optimizer_args
from predict import parse_predict_args
from trainer import parse_trainer_args
from utils import _ARGPARSE_HANDLERS

_KWARG_TYPE = dict[str, Any]


@dataclass
class Args:
    model: _KWARG_TYPE
    loss: _KWARG_TYPE
    augmentation: _KWARG_TYPE
    data: _KWARG_TYPE
    trainer: _KWARG_TYPE
    experiment: _KWARG_TYPE
    optimizer: _KWARG_TYPE
    predict: _KWARG_TYPE
    mode: Literal["train", "test", "predict"]

    @classmethod
    def from_argparse(cls, args: argparse.Namespace):
        return cls(
            model=parse_model_args(args),
            loss=parse_loss_args(args),
            augmentation=parse_augmentation_args(args),
            data=parse_data_args(args),
            trainer=parse_trainer_args(args),
            experiment=parse_experiment_args(args),
            optimizer=parse_optimizer_args(args),
            predict=parse_predict_args(args),
            mode=args.mode,
        )


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train or predict genome-level embeddings based on sets of protein-level embeddings"
    )
    parser.add_argument(
        "--mode",
        metavar="",
        choices={"train", "predict", "test"},
        default="train",
        help="model mode (default: %(default)s) [choices: %(choices)s]",
    )

    for handler in _ARGPARSE_HANDLERS:
        handler(parser)

    args = parser.parse_args()
    return Args.from_argparse(args)
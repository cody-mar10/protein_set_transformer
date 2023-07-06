from __future__ import annotations

from typing import Any, cast

import lightning as L
import torch

from pst import CrossValidationTrainer, FullTrainer, Predictor, training
from pst.utils.cli.pydantic import (
    Args,
    InferenceMode,
    TrainingMode,
    convert,
    parse_args,
)

_SEED = 111

SerializedModelT = dict[str, dict[str, Any]]


def train_main(mode: TrainingMode):
    args = cast(SerializedModelT, mode.dict(exclude={"predict"}))
    tune = args["experiment"].pop("tune", True)
    n_trials = args["experiment"].pop("n_trials")
    prune = args["experiment"].pop("prune", True)
    parallel = args["experiment"].pop("parallel")
    config = args["experiment"].pop("config", None)

    if tune:
        # flatten args, ie remove organization
        cv_trainer_kwargs = {
            key: value for kwargs in args.values() for key, value in kwargs.items()
        }
        training.optimize(
            n_trials=n_trials,
            prune=prune,
            parallel=parallel,
            configfile=config,
            **cv_trainer_kwargs,
        )
    elif args["data"]["train_on_full"]:
        trainer = FullTrainer.from_kwargs(**args)
        trainer.train()
    else:
        L.seed_everything(_SEED)
        trainer = CrossValidationTrainer.from_kwargs(**args)
        avg_val_loss = trainer.train_with_cross_validation()
        print(f"Average validation loss: {avg_val_loss}")


def test_main(args: Args):
    raise NotImplementedError


def predict_main(mode: InferenceMode):
    args: SerializedModelT = mode.dict(include={"predict", "trainer", "data"})
    L.seed_everything(_SEED)
    predictor = Predictor(**args)
    predictor.predict()


def main():
    args = parse_args()
    is_training = args.train is not None
    is_predicting = args.predict is not None
    args = convert(args)

    if args.trainer.accelerator == "cpu":
        threads = args.trainer.devices
        args.trainer.precision = "32"
        torch.set_num_threads(threads)
        args.trainer.devices = 1
        args.trainer.strategy = "auto"

    if is_training:
        train_main(args)
    elif is_predicting:
        predict_main(args)  # type: ignore
    else:
        test_main(args)  # type: ignore


if __name__ == "__main__":
    main()

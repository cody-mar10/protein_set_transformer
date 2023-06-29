from __future__ import annotations

import lightning as L
import torch

import pst
from pst import CrossValidationTrainer, Predictor
from pst.utils.cli import Args, parse_args

_SEED = 111


def train_main(args: Args):
    tune = args.experiment.pop("tune", True)
    n_trials = args.experiment.pop("n_trials")
    prune = args.experiment.pop("prune", True)
    parallel = args.experiment.pop("parallel")
    if tune:
        cv_trainer_kwargs = args.flatten(ignore="predict")
        pst.training.optimize(
            n_trials=n_trials, prune=prune, parallel=parallel, **cv_trainer_kwargs
        )
    else:
        L.seed_everything(_SEED)
        trainer = CrossValidationTrainer.from_cli_args(args)
        avg_val_loss = trainer.train_with_cross_validation()
        print(f"Average validation loss: {avg_val_loss}")


def test_main(args: Args):
    raise NotImplementedError


def predict_main(args: Args):
    L.seed_everything(_SEED)
    predictor = Predictor.from_cli_args(args)
    predictor.predict()


def main():
    args = parse_args()

    if args.trainer["accelerator"] == "cpu":
        threads = args.trainer["devices"]
        args.trainer["precision"] = 32
        torch.set_num_threads(threads)
        args.trainer["devices"] = 1
        args.trainer["strategy"] = "auto"
    elif args.trainer["accelerator"] == "gpu":
        args.trainer["num_nodes"] = 1

    if args.mode == "train":
        train_main(args)
    elif args.mode == "predict":
        predict_main(args)
    else:
        test_main(args)


if __name__ == "__main__":
    main()

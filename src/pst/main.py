from __future__ import annotations

import lightning as L
import torch

import pst
from pst.utils.cli import (
    Args,
    InferenceMode,
    TrainingMode,
    parse_args,
)

_SEED = 111


def train_main(args: TrainingMode):
    if args.experiment.tune:
        pst.training.tuning.tune(args)
    elif args.data.train_on_full:
        pst.training.full.train_with_all_data(args)
    else:
        L.seed_everything(_SEED)
        pst.training.cv.train_with_cross_validation(args)


def test_main(args: Args):
    raise NotImplementedError


def predict_main(args: InferenceMode):
    L.seed_everything(_SEED)
    predictor = pst.predict.Predictor(
        **args.model_dump(include={"predict", "trainer", "data"})
    )
    predictor.predict()


def _check_cpu_accelerator(config: TrainingMode | InferenceMode):
    if config.trainer.accelerator == "cpu":
        threads = config.trainer.devices
        config.trainer.precision = "32"
        torch.set_num_threads(threads)
        config.trainer.devices = 1
        config.trainer.strategy = "auto"


def main():
    args = parse_args()

    if args.train is not None:
        config = args.train
        _check_cpu_accelerator(config)
        train_main(config)
    elif args.predict is not None:
        config = args.predict
        _check_cpu_accelerator(config)
        predict_main(config)
    else:
        raise RuntimeError("Must pass either 'train' or 'predict' as running mode.")


if __name__ == "__main__":
    main()

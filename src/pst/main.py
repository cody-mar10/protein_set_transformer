from __future__ import annotations

import lightning as L
import torch

from pst import Predictor, Trainer
from pst.utils.cli import Args, parse_args


def train_main(args: Args):
    trainer = Trainer.from_cli_args(args)
    trainer.train_with_cross_validation()


def test_main(args: Args):
    raise NotImplementedError


def predict_main(args: Args):
    predictor = Predictor.from_cli_args(args)
    predictor.predict()


def main():
    args = parse_args()
    L.seed_everything(111)
    print("Lightning version: ", L.__version__)

    if args.trainer["accelerator"] == "cpu":
        threads = args.trainer["devices"]
        args.trainer["precision"] = 32
        torch.set_num_threads(threads)
        args.trainer["devices"] = 1
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

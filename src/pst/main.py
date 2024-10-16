from __future__ import annotations

import logging

import lightning as L
import torch
from torch.autograd.anomaly_mode import set_detect_anomaly

from pst import _logger as logger
from pst.embed import embed
from pst.predict import model_inference
from pst.training import cv, finetune, full
from pst.training.tuning import tuning
from pst.utils.cli import Args, parse_args
from pst.utils.cli.modes import FinetuningMode, InferenceMode, TrainingMode, TuningMode
from pst.utils.download import download
from pst.utils.graphify import to_graph_format
from pst.utils.history import update_config_from_history

_SEED = 111
# logger = logging.getLogger(__name__)


def train_main(args: TrainingMode):
    args = update_config_from_history(args)
    if args.data.train_on_full:
        full.train_with_all_data(args)
    else:
        L.seed_everything(_SEED)
        cv.train_with_cross_validation(args)


def tune_main(args: TuningMode):
    # this is for hyperparameter tuning
    tuning.tune(args)


def test_main(args: Args):
    raise NotImplementedError


def predict_main(args: InferenceMode):
    L.seed_everything(_SEED)
    model_inference(args, True, True, False)


def finetune_main(args: FinetuningMode):
    # set lightning trainer dir to --outdir
    args.trainer.default_root_dir = args.finetuning.outdir
    finetune.finetune(args)


def _check_cpu_accelerator(config: TrainingMode):
    if config.trainer.accelerator == "cpu":
        threads = config.trainer.devices
        config.trainer.precision = "32"
        torch.set_num_threads(threads)
        config.trainer.devices = 1
        config.trainer.strategy = "auto"


def _validate_accelerator(config: TrainingMode):
    if config.trainer.accelerator == "auto":
        config.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"


def _setup_logger():
    logger.setLevel(logging.INFO)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] (%(levelname)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)


def main():
    args = parse_args()
    _setup_logger()

    if args.graphify is not None:
        to_graph_format(args.graphify.graphify)
        return
    elif args.download is not None:
        download(args.download.download)
        return
    elif args.embed is not None:
        embed(args.embed.embed)
        return
    elif args.train is not None:
        config = args.train
        fn = train_main
    elif args.tune is not None:
        config = args.tune
        fn = tune_main
    elif args.predict is not None:
        config = args.predict
        fn = predict_main
    elif args.finetune is not None:
        config = args.finetune
        fn = finetune_main
    else:
        # should not really get here since argparsing should already catch this
        raise RuntimeError("Invalid run mode passed.")

    _validate_accelerator(config)
    _check_cpu_accelerator(config)

    with set_detect_anomaly(config.experiment.detect_anomaly):
        if config.experiment.detect_anomaly:
            logger.warning(
                "Anomaly detection mode is on. This will be slow since the autograd "
                "engine has to check for gradient anomalies."
            )

        fn(config)  # type: ignore


if __name__ == "__main__":
    main()

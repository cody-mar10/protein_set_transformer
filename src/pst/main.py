#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import lightning as L
from lightning.pytorch.cli import LightningCLI

import pst

# TODO: put arg parsing into separate file, prob utils

AcceleratorChoices = Literal["cpu", "gpu", "tpu", "auto"]


@dataclass
class TrainerHyperparameters:
    accelerator: AcceleratorChoices
    devices: int
    default_root_dir: Path
    max_epochs: int


@dataclass
class OptimHyperparameters:
    lr: float
    weight_decay: float


@dataclass
class ModelHyperparameters:
    n_heads: int
    scale: float
    alpha: float


@dataclass
class Args:
    input: Path
    trainer_hparams: dict[str, Any]
    optim_hparams: dict[str, Any]
    model_hparams: dict[str, Any]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train or predict genome-level embeddings based on sets of protein-level embeddings"
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        type=Path,
        required=True,
        help="input protein embeddings file in .h5 format",
    )
    args = parser.parse_args()
    trainer_hparams = asdict(
        TrainerHyperparameters(
            accelerator="auto",
            devices=4,
            default_root_dir=Path("checkpoints"),
            max_epochs=1000,
        )
    )
    optim_hparams = asdict(OptimHyperparameters(lr=1e-3, weight_decay=0.01))
    model_hparams = asdict(ModelHyperparameters(4, 7.0, 0.1))
    return Args(
        input=args.input,
        trainer_hparams=trainer_hparams,
        optim_hparams=optim_hparams,
        model_hparams=model_hparams,
    )


def main():
    cli = LightningCLI(
        model_class=pst.modules.GenomeClassifier,
        datamodule_class=pst.data.GenomeSetDataModule,
        seed_everything_default=111,
    )

    # args = parse_args()
    # trainer = L.Trainer(**args.trainer_hparams)


if __name__ == "__main__":
    main()

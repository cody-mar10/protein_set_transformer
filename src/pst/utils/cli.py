from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional

_ADDER_TYPE = Callable[[argparse.ArgumentParser], None]
_ARGPARSE_HANDLERS: list[_ADDER_TYPE] = list()


def register(func: _ADDER_TYPE):
    _ARGPARSE_HANDLERS.append(func)


@dataclass
class TrainerArgs:
    devices: int
    accelerator: Literal["cpu", "gpu", "tpu", "auto"]
    # default_root_dir: Path
    max_epochs: int


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    patience: int


@dataclass
class ModelArgs:
    # skip in_dim
    out_dim: int
    hidden_dim: int
    num_heads: int
    n_outputs: int
    num_indices: int
    n_enc_layers: int
    n_dec_layers: int
    dropout: float
    bias: bool
    norm: bool
    sample_scale: float
    sample_rate: float
    loss_alpha: float


@dataclass
class DataArgs:
    data_file: Path
    metadata_file: Path
    batch_size: int
    split_ratio: Optional[tuple[float, ...]]
    num_workers: int
    pin_memory: bool


@dataclass
class LoggerArgs:
    root_dir: Path
    name: str
    flush_logs_every_n_steps: int


@dataclass
class Args:
    model: dict[str, Any]
    data: dict[str, Any]
    trainer: dict[str, Any]
    optimizer: dict[str, Any]
    logger: dict[str, Any]


@register
def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("DATA ARGS")
    group.add_argument(
        "-d",
        "--data_file",
        metavar="FILE",
        type=Path,
        required=True,
        help="input protein embeddings file in .h5 format",
    )
    group.add_argument(
        "-m",
        "--metadata_file",
        metavar="FILE",
        type=Path,
        required=True,
        help="metadata file",
    )
    group.add_argument(
        "-b",
        "--batch_size",
        metavar="INT",
        type=int,
        default=128,
        help="batch size (default: %(default)s)",
    )
    group.add_argument(
        "--split_ratio",
        metavar="[FLOAT FLOAT]",
        nargs=2,
        type=float,
        default=(0.8, 0.2),
        help="train/val split ratios (default: %(default)s)",
    )
    group.add_argument(
        "--num_workers",
        metavar="INT",
        type=int,
        default=0,
        help="additional cpu workers to load data (default: %(default)s)",
    )
    group.add_argument(
        "--no-pin_memory",
        action="store_true",
        help="whether to pin memory onto a CUDA GPU (default: %(default)s)",
    )


def parse_data_args(args: argparse.Namespace) -> DataArgs:
    return DataArgs(
        data_file=args.data_file,
        metadata_file=args.metadata_file,
        batch_size=args.batch_size,
        split_ratio=tuple(args.split_ratio),
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )


@register
def add_model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("MODEL ARGS")
    group.add_argument(
        "--out_dim",
        metavar="INT",
        type=int,
        default=64,
        help="output dimension (default: %(default)s)",
    )
    group.add_argument(
        "--hidden_dim",
        metavar="INT",
        type=int,
        default=128,
        help="hidden layer dimension (default: %(default)s)",
    )
    group.add_argument(
        "--num_heads",
        metavar="INT",
        type=int,
        default=4,
        help="number of attention heads (default: %(default)s)",
    )
    group.add_argument(
        "--n_outputs",
        metavar="INT",
        type=int,
        default=1,
        help="number of model outputs per genome (default: %(default)s)",
    )
    group.add_argument(
        "--num_indices",
        metavar="INT",
        type=int,
        default=32,
        help="number of projection indices for efficient large-set pairwise attention (default: %(default)s)",
    )
    group.add_argument(
        "--n_enc_layers",
        metavar="INT",
        type=int,
        default=2,
        help="number of encoder layers (default: %(default)s)",
    )
    group.add_argument(
        "--n_dec_layers",
        metavar="INT",
        type=int,
        default=2,
        help="number of decoder layers (not including first pooled attention layer and any subsequent fully connected layers) (default: %(default)s)",
    )
    group.add_argument(
        "--dropout",
        metavar="FLOAT",
        type=float,
        default=0.0,
        help="dropout proportion during training (default: %(default)s)",
    )
    # ignoring bias and norm -> will be set to always on for now
    group.add_argument(
        "--sample_scale",
        metavar="FLOAT",
        type=float,
        default=7.0,
        help="exponential decay scale factor for weighting negative samples during loss (default: %(default)s)",
    )
    group.add_argument(
        "--sample_rate",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help="PointSwap sampler swapping rate (default: %(default)s)",
    )
    group.add_argument(
        "--loss_alpha",
        metavar="FLOAT",
        type=float,
        default=0.1,
        help="constant term in loss function (default: %(default)s)",
    )


def parse_model_args(args: argparse.Namespace) -> ModelArgs:
    return ModelArgs(
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        n_outputs=args.n_outputs,
        num_indices=args.num_indices,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        dropout=args.dropout,
        bias=True,
        norm=True,
        sample_scale=args.sample_scale,
        sample_rate=args.sample_rate,
        loss_alpha=args.loss_alpha,
    )


@register
def add_trainer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("TRAINER ARGS")
    group.add_argument(
        "--devices",
        metavar="INT",
        type=int,
        default=1,
        help="number of accelerator devices to use. For CPUs, this sets the total thread usage. (default: %(default)s)",
    )
    group.add_argument(
        "--accelerator",
        metavar="DEVICE",
        choices={"cpu", "gpu", "tpu", "auto"},
        default="gpu",
        help="accelerator to use (default: %(default)s)",
    )
    # group.add_argument(
    #     "--default_root_dir",
    #     metavar="DIR",
    #     type=Path,
    #     default=Path.cwd().joinpath("lightning_root"),
    #     help="lightning root dir for model checkpointing (default: %(default)s)",
    # )
    group.add_argument(
        "--max_epochs",
        metavar="INT",
        type=int,
        default=1000,
        help="max number of training epochs (default: %(default)s)",
    )


def parse_trainer_args(args: argparse.Namespace) -> TrainerArgs:
    return TrainerArgs(
        devices=args.devices,
        accelerator=args.accelerator,
        # default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
    )


@register
def add_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("OPTIMIZER ARGS")
    group.add_argument(
        "--lr",
        metavar="FLOAT",
        type=float,
        default=1e-3,
        help="learning rate (default: %(default)s)",
    )
    group.add_argument(
        "--weight_decay",
        metavar="FLOAT",
        type=float,
        default=0.0,
        help="optimizer weight decay (default: %(default)s)",
    )
    group.add_argument(
        "--patience",
        metavar="int",
        type=int,
        default=5,
        help="number of epochs for plateau learning rate scheduler to wait for a stagnating training session to reduce the learning rate (default: %(default)s)",
    )
    group.add_argument(
        "--betas",
        metavar="[FLOAT FLOAT]",
        type=float,
        nargs=2,
        default=(0.9, 0.999),
        help="optimizer betas (default: %(default)s)",
    )


def parse_optimizer_args(args: argparse.Namespace) -> OptimizerArgs:
    return OptimizerArgs(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        patience=args.patience,
    )


@register
def add_logger_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("LOGGER ARGS")
    group.add_argument(
        "--root_dir",
        metavar="DIR",
        type=Path,
        default=Path.cwd().joinpath("lightning_logs"),
        help="logging directory for model checkpointing (default: %(default)s)",
    )
    group.add_argument(
        "--name",
        metavar="STR",
        default="genome-transformer",
        help="experiment name (default: %(default)s)",
    )
    group.add_argument(
        "--logging-rate",
        metavar="INT",
        type=int,
        default=10,
        help="number of epochs to flush the in-memory log to disk (default: %(default)s)",
    )


def parse_logging_args(args: argparse.Namespace) -> LoggerArgs:
    return LoggerArgs(
        root_dir=args.root_dir,
        name=args.name,
        flush_logs_every_n_steps=args.logging_rate,
    )


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train or predict genome-level embeddings based on sets of protein-level embeddings"
    )

    for handler in _ARGPARSE_HANDLERS:
        handler(parser)

    args = parser.parse_args()
    return Args(
        model=asdict(parse_model_args(args)),
        data=asdict(parse_data_args(args)),
        trainer=asdict(parse_trainer_args(args)),
        optimizer=asdict(parse_optimizer_args(args)),
        logger=asdict(parse_logging_args(args)),
    )

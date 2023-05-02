from __future__ import annotations

import argparse
import sys
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
    default_root_dir: Path
    max_epochs: int
    precision: Literal["16-mixed", "bf16-mixed", 32, "32"]
    strategy: Literal["ddp", "ddp_spawn"]


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    warmup_steps: int


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
    loss_margin: float
    compile: bool


@dataclass
class DataArgs:
    data_file: Path
    metadata_file: Path
    batch_size: int
    split_ratio: Optional[tuple[float, ...]]
    num_workers: int
    pin_memory: bool


@dataclass
class PredictArgs:
    checkpoint: Path
    outdir: Path


@dataclass
class Args:
    model: dict[str, Any]
    data: dict[str, Any]
    trainer: dict[str, Any]
    optimizer: dict[str, Any]
    predict: dict[str, Any]
    mode: Literal["train", "test", "predict", "debug", "precompute"]


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
        "--loss_margin",
        metavar="FLOAT",
        type=float,
        default=0.1,
        help="triplet loss margin (default: %(default)s)",
    )
    group.add_argument(
        "--compile",
        action="store_true",
        help="compile model using torch.compile (default: %(default)s)",
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
        loss_margin=args.loss_margin,
        compile=args.compile,
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
    group.add_argument(
        "--default_root_dir",
        metavar="DIR",
        type=Path,
        default=Path.cwd().joinpath("lightning_root"),
        help="lightning root dir for model checkpointing (default: %(default)s)",
    )
    group.add_argument(
        "--max_epochs",
        metavar="INT",
        type=int,
        default=1000,
        help="max number of training epochs (default: %(default)s)",
    )
    group.add_argument(
        "--strategy",
        metavar="",
        choices={"ddp", "ddp_spawn"},
        default="ddp_spawn",
        help="parallelized training strategy (default: %(default)s) [choices: %(choices)s]",
    )
    group.add_argument(
        "--precision",
        metavar="",
        choices={"16-mixed", 32, "bf16-mixed"},
        default="16-mixed",
        type=lambda x: int(x) if x.isdigit() else x,
        help="floating point precision (default: %(default)s) [choices: %(choices)s]",
    )


def parse_trainer_args(args: argparse.Namespace) -> TrainerArgs:
    return TrainerArgs(
        devices=args.devices,
        accelerator=args.accelerator,
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        precision=args.precision,
        strategy=args.strategy,
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
        "--warmup_steps",
        metavar="int",
        type=int,
        default=5000,
        help="number of training steps to warm-up the learning rate to the max (default: %(default)s)",
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
        warmup_steps=args.warmup_steps,
    )


@register
def add_predict_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("PREDICT ARGS")
    is_inference_mode = "predict" in sys.argv
    group.add_argument(
        "--checkpoint",
        metavar="FILE",
        type=Path,
        required=is_inference_mode,
        help="required model checkpoint during inference",
    )
    group.add_argument(
        "--outdir",
        metavar="DIR",
        type=Path,
        default=Path.cwd().joinpath("output"),
        help="inference output directory (default: %(default)s)",
    )


def parse_predict_args(args: argparse.Namespace) -> PredictArgs:
    if args.mode == "predict" and args.checkpoint is None:
        raise ValueError(
            "During inference (predict) mode, a model checkpoint must be supplied."
        )

    return PredictArgs(
        checkpoint=args.checkpoint,
        outdir=args.outdir,
    )


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train or predict genome-level embeddings based on sets of protein-level embeddings"
    )
    parser.add_argument(
        "--mode",
        metavar="",
        choices={"train", "predict", "test", "debug"},
        default="train",
        help="model mode (default: %(default)s) [choices: %(choices)s]",
    )

    for handler in _ARGPARSE_HANDLERS:
        handler(parser)

    args = parser.parse_args()
    return Args(
        model=asdict(parse_model_args(args)),
        data=asdict(parse_data_args(args)),
        trainer=asdict(parse_trainer_args(args)),
        optimizer=asdict(parse_optimizer_args(args)),
        predict=asdict(parse_predict_args(args)),
        mode=args.mode,
    )

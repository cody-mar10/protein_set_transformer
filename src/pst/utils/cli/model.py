from __future__ import annotations

import argparse
from dataclasses import dataclass

from .utils import asdict, register_defaults, register_handler


@dataclass
class ModelArgs:
    # skip in_dim
    out_dim: int = -1
    num_heads: int = 4
    n_enc_layers: int = 5
    multiplier: float = 1.0
    dropout: float = 0.5
    compile: bool = False


_DEFAULTS = ModelArgs()
register_defaults(_DEFAULTS, "model")


@register_handler
def add_model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("MODEL ARGS")
    group.add_argument(
        "--out_dim",
        metavar="INT",
        type=int,
        default=_DEFAULTS.out_dim,
        help=(
            "output dimension (default: %(default)s which means use the input "
            "embedding dimension)"
        ),
    )
    group.add_argument(
        "--num_heads",
        metavar="INT",
        type=int,
        default=_DEFAULTS.num_heads,
        help="number of attention heads (default: %(default)s)",
    )
    group.add_argument(
        "--n_enc_layers",
        metavar="INT",
        type=int,
        default=_DEFAULTS.n_enc_layers,
        help="number of encoder layers (default: %(default)s)",
    )
    group.add_argument(
        "--multiplier",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.multiplier,
        help=(
            "multiplicative weight to de-emphasize (< 1.0) or over-emphasize (> 1.0) "
            "protein weights when decoding a genome representation by pooling over "
            "all proteins in a genome (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--dropout",
        metavar="FLOAT",
        type=float,
        default=_DEFAULTS.dropout,
        help="dropout proportion during training (default: %(default)s)",
    )
    group.add_argument(
        "--compile",
        action="store_true",
        help="compile model using torch.compile (default: %(default)s)",
    )


@asdict
def parse_model_args(args: argparse.Namespace) -> ModelArgs:
    return ModelArgs(
        out_dim=args.out_dim,
        num_heads=args.num_heads,
        n_enc_layers=args.n_enc_layers,
        multiplier=args.multiplier,
        dropout=args.dropout,
        compile=args.compile,
    )

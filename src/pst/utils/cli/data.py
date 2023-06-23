from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .utils import _NONEXISTENT_FILE, asdict, register


@dataclass
class DataArgs:
    data_file: Path
    batch_size: int = 32
    train_on_full: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    edge_strategy: Literal["chunked", "sparse", "full"] = "chunked"
    chunk_size: Optional[int] = 30
    threshold: Optional[int] = -1


_DEFAULTS = DataArgs(data_file=_NONEXISTENT_FILE)


@register
def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("DATA ARGS")
    group.add_argument(
        "-d",
        "--data_file",
        metavar="FILE",
        type=Path,
        required=True,
        help="input protein embeddings file in .h5 file format with the fields .data for protein embeddings",  # TODO: add more desc  # noqa: E501
    )
    group.add_argument(
        "-b",
        "--batch_size",
        metavar="INT",
        type=int,
        default=_DEFAULTS.batch_size,
        help="batch size (default: %(default)s)",
    )
    group.add_argument(
        "--num_workers",
        metavar="INT",
        type=int,
        default=_DEFAULTS.num_workers,
        help="additional cpu workers to load data (default: %(default)s)",
    )
    group.add_argument(
        "--no-pin_memory",
        action="store_true",
        help="whether to pin memory onto a CUDA GPU (default: %(default)s)",
    )
    group.add_argument(
        "--train-on-full",
        action="store_true",
        help=(
            "whether to train a single model on the full input data. When not "
            "specified, the default is  to train multiple models with cross "
            "validation (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--edge_strategy",
        metavar="",
        choices={"chunked", "full", "sparse"},
        default=_DEFAULTS.edge_strategy,
        help=(
            "strategy to create 'edges' between protein nodes in a genome graph. "
            "chunked = split genomes in --chunk_size chunks. sparse = remove "
            "interactions longer than --threshold. full = fully connected graph "
            "like a regular transformer. (default: %(default)s) [choices: %(choices)s]"
        ),
    )
    group.add_argument(
        "--chunk_size",
        metavar="INT",
        type=int,
        default=_DEFAULTS.chunk_size,
        help=(
            "size of sub-chunks to break genomes into if using --edge_strategy "
            "chunked (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--threshold",
        metavar="INT",
        type=int,
        default=_DEFAULTS.threshold,
        help=(
            "range of protein interactions if using --edge_strategy [chunked|sparse] "
            "(default: %(default)s which means don't prune by distance)"
        ),
    )


@asdict
def parse_data_args(args: argparse.Namespace) -> DataArgs:
    return DataArgs(
        data_file=args.data_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        train_on_full=args.train_on_full,
        edge_strategy=args.edge_strategy,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
    )

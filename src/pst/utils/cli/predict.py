from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from utils import asdict, register, _NONEXISTENT_FILE


@dataclass
class PredictArgs:
    checkpoint: Path
    outdir: Path = Path("output")


_DEFAULTS = PredictArgs(checkpoint=_NONEXISTENT_FILE)


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
        default=_DEFAULTS.outdir,
        help="inference output directory (default: %(default)s)",
    )


@asdict
def parse_predict_args(args: argparse.Namespace) -> PredictArgs:
    if args.mode == "predict" and args.checkpoint is None:
        raise ValueError(
            "During inference (predict) mode, a model checkpoint must be supplied."
        )

    return PredictArgs(
        checkpoint=args.checkpoint,
        outdir=args.outdir,
    )

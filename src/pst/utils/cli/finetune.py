from __future__ import annotations

from pathlib import Path

from pydantic import Field

from pst.utils.cli.predict import PredictArgs


# basically shares same fields with PredictArgs
# but need to change descriptions
class FinetuningArgs(PredictArgs):
    checkpoint: Path = Field(..., description="pre-trained model checkpoint")
    outdir: Path = Field(Path("output"), description="finetuning output directory")

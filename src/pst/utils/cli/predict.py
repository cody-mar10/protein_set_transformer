from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PredictArgs(BaseModel):
    checkpoint: Path = Field(..., description="model checkpoint during inference")
    outdir: Path = Field(Path("output"), description="inference output directory")
    fragment_oversized_genomes: bool = Field(
        False,
        description="fragment oversized genomes that encode more proteins than the model was trained to expect",
    )

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PredictArgs(BaseModel):
    checkpoint: Path = Field(..., description="model checkpoint during inference")
    outdir: Path = Field(Path("output"), description="inference output directory")

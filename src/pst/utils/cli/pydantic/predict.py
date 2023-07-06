from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .utils import register_model

_NAME = "predict"


@register_model(_NAME)
class PredictArgs(BaseModel):
    checkpoint: Path = Field(
        ..., description="required model checkpoint during inference"
    )
    outdir: Path = Field(Path("output"), description="inference output directory")

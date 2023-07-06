from __future__ import annotations

from pydantic import BaseModel, Field

from .utils import register_model

_NAME = "loss"


@register_model(_NAME)
class LossArgs(BaseModel):
    margin: float = Field(0.1, description="triplet loss margin", gt=0.0)

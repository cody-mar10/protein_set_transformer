from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .utils import register_model

NO_NEGATIVES_MODES = Literal["closest_to_positive", "closest_to_anchor"]

_NAME = "augmentation"


@register_model(_NAME)
class AugmentationArgs(BaseModel):
    sample_scale: float = Field(
        7.0,
        description="exponential decay scale factor for weighting negative samples during loss",  # noqa: E501
        gt=0.0,
    )
    sample_rate: float = Field(
        0.5, description="PointSwap sampler swapping rate", gt=0.0, lt=1.0
    )
    no_negatives_mode: NO_NEGATIVES_MODES = Field(
        "closest_to_positive",
        description="mode to handle event of no semihard negative sample existing",
    )

from __future__ import annotations

from pydantic import BaseModel, Field

from .utils import register_model

_NAME = "model"


@register_model(_NAME)
class ModelArgs(BaseModel):
    out_dim: int = Field(
        -1, description="output dimension, default is to use the input dimension"
    )
    num_heads: int = Field(4, description="number of attention heads", gt=0)
    n_enc_layers: int = Field(5, description="number of encoder layers", gt=0)
    multiplier: float = Field(
        1.0,
        description=(
            "multiplicative weight to de-emphasize (< 1.0) or over-emphasize "
            "(> 1.0) protein weights when decoding a genome representation by "
            "pooling over all proteins in a genome"
        ),
        gt=0.0,
    )
    dropout: float = Field(
        0.5, description="dropout proportion during training", ge=0.0, lt=1.0
    )
    compile: bool = Field(False, description="compile model using torch.compile")

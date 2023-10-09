from __future__ import annotations

from lightning_cv import BaseModelConfig
from pydantic import BaseModel, Field

from pst.typing import NO_NEGATIVES_MODES


class AugmentationConfig(BaseModel):
    sample_scale: float = Field(
        7.0,
        description=(
            "exponential decay scale factor for weighting negative samples during loss"
        ),
        gt=0.0,
    )
    sample_rate: float = Field(
        0.5, description="PointSwap sampler swapping rate", gt=0.0, lt=1.0
    )
    no_negatives_mode: NO_NEGATIVES_MODES = Field(
        "closest_to_positive",
        description="mode to handle event of no semihard negative sample existing",
    )


class LossConfig(BaseModel):
    margin: float = Field(0.1, description="triplet loss margin", gt=0.0)


class OptimizerConfig(BaseModel):
    lr: float = Field(1e-3, description="learning rate", ge=1e-5, le=1e-1)
    weight_decay: float = Field(
        0.0, description="optimizer weight decay", ge=0.0, le=1e-1
    )
    betas: tuple[float, float] = Field((0.9, 0.999), description="optimizer betas")
    warmup_steps: int = Field(0, description="number of warmup steps", ge=0)
    use_scheduler: bool = Field(
        False, description="whether or not to use a linearly decaying scheduler"
    )


class ModelConfig(BaseModelConfig):
    in_dim: int = Field(
        -1, description="input dimension, default is to use the dataset dimension"
    )
    out_dim: int = Field(
        -1, description="output dimension, default is to use the input dimension"
    )
    num_heads: int = Field(4, description="number of attention heads", gt=0)
    n_enc_layers: int = Field(5, description="number of encoder layers", gt=0)
    embed_scale: int = Field(
        4,
        description=(
            "scale factor for positional and strand embeddings. These embeddings will be "
            "of size in_dim/n, ie higher number means smaller embeddings."
        ),
        ge=1,
        le=8,
    )
    dropout: float = Field(
        0.5, description="dropout proportion for individual weights", ge=0.0, lt=1.0
    )
    layer_dropout: float = Field(
        0.0, description="dropout proportion for entire layers", ge=0.0, lt=1.0
    )
    compile: bool = Field(False, description="compile model using torch.compile")
    optimizer: OptimizerConfig
    loss: LossConfig
    augmentation: AugmentationConfig

    @classmethod
    def default(cls):
        schema = dict()
        for key, field in cls.model_fields.items():
            if isinstance(field.annotation, type):
                if isinstance(field.annotation, BaseModel) or issubclass(
                    field.annotation, BaseModel
                ):
                    # field is a pydantic model
                    # NOTE: does not handle any nesting below this...
                    value = field.annotation.model_construct()
                else:
                    value = field.get_default()

                schema[key] = value

        return cls.model_validate(schema)

from __future__ import annotations

from lightning_cv import BaseModelConfig as _BaseModelConfig
from pydantic import BaseModel, Field

from pst.typing import NO_NEGATIVES_MODES

MAX_PROTEINS_PER_GENOME = 2048


class AugmentationConfig(BaseModel):
    sample_rate: float = Field(
        0.5, description="PointSwap sampler swapping rate", gt=0.0, lt=1.0
    )


class BaseLossConfig(BaseModel):
    """This is used to pass arguments to setting up the loss function.

    Subclass this if you need to pass additional arguments to the loss function.
    """

    pass


class GenomeTripletLossConfig(BaseLossConfig):
    margin: float = Field(0.1, description="triplet loss margin", gt=0.0)
    sample_scale: float = Field(
        7.0,
        description=(
            "exponential decay scale factor for weighting negative samples during triplet loss"
        ),
        gt=0.0,
    )
    no_negatives_mode: NO_NEGATIVES_MODES = Field(
        "closest_to_positive",
        description="mode to handle event of no semihard negative sample existing",
    )


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


class BaseModelConfig(_BaseModelConfig):
    """Base config for all ProteinSetTransformer models.

    This can be used as is, but subclassing can be used to add additional parameters.
    Additionally, a custom loss config can be passed to the loss field when subclassing to
    overwrite the default loss config.
    """

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
    proj_cat: bool = Field(
        False,
        description=(
            "whether to project the concatenated pLM, positional, and strand embeddings "
            "back to the original dimensionality"
        ),
    )
    max_proteins: int = Field(
        MAX_PROTEINS_PER_GENOME,
        description=(
            "maximum number of proteins per genome. This affects the possible size of the "
            "positional embeddings and does not need to be the actual maximum number of "
            "proteins in the dataset. The actual maximum number of proteins just needs to be "
            "less than or equal to this value. Increasing this value will lead to increases in "
            "model size equal to this value * embedding_dim / embed_scale * 4 bytes."
        ),
    )
    compile: bool = Field(False, description="compile model using torch.compile")
    optimizer: OptimizerConfig
    loss: BaseLossConfig
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


class ModelConfig(BaseModelConfig):
    loss: GenomeTripletLossConfig

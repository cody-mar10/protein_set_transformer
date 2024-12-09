from dataclasses import dataclass, field
from typing import Literal, Union

from jsonargparse.typing import (
    NonNegativeInt,
    OpenUnitInterval,
    PositiveFloat,
    PositiveInt,
    restricted_number_type,
)

from pst.typing import NO_NEGATIVES_MODES
from pst.utils.dataclass_utils import DataclassValidatorMixin, validated_field

MAX_PROTEINS_PER_GENOME = 2048


@dataclass
class BaseAugmentationConfig(DataclassValidatorMixin):
    """This is used to pass arguments to setting up the augmentation function.

    Subclass this if you need to pass additional arguments to the augmentation function.
    """

    pass


@dataclass
class AugmentationConfig(BaseAugmentationConfig):
    """AUGMENTATION

    using PointSwap.

    Attributes:
        sample_rate (float): PointSwap sampler swapping rate in (0.0, 1.0)
    """

    sample_rate: float = validated_field(0.5, validator=OpenUnitInterval)
    """PointSwap sampler swapping rate in (0.0, 1.0)"""


@dataclass
class BaseLossConfig(DataclassValidatorMixin):
    """This is used to pass arguments to setting up the loss function.

    Subclass this if you need to pass additional arguments to the loss function.
    """

    pass


@dataclass
class WeightedLossConfig(BaseLossConfig):
    """Base config for weighted loss functions that need a weight scale factor.

    Attributes:
        sample_scale (float): exponential decay scale factor for weighting samples during triplet, contrastive, or relational loss
    """

    sample_scale: float = validated_field(7.0, validator=PositiveFloat)
    """exponential decay scale factor for weighting samples during triplet, contrastive, or 
    relational loss"""


@dataclass
class GenomeTripletLossConfig(WeightedLossConfig):
    """LOSS

    Triplet loss with exponential decay weighting.

    Attributes:
        margin (float): triplet loss margin
        no_negatives_mode (NO_NEGATIVES_MODES): mode to handle event of no semihard negative
            sample existing
    """

    margin: float = validated_field(0.1, validator=PositiveFloat)
    """triplet loss margin"""

    no_negatives_mode: NO_NEGATIVES_MODES = "closest_to_positive"
    """mode to handle event of no semihard negative sample existing"""


LearningRate = restricted_number_type(
    "LearningRate",
    float,
    [(">=", 1e-5), ("<=", 1e-1)],
    docstring="learning rate in [1e-5, 1e-1]",
)

WeightDecay = restricted_number_type(
    "WeightDecay",
    float,
    [(">=", 0.0), ("<=", 1e-1)],
    docstring="optimizer weight decay in [0.0, 1e-1]",
)


@dataclass
class OptimizerConfig(DataclassValidatorMixin):
    """OPTIMIZER
    for constructing a `torch.optim.AdamW` optimizer.

    Attributes:
        lr (float): learning rate
        weight_decay (float): optimizer weight decay
        betas (tuple[float, float]): optimizer betas
        warmup_steps (int): number of warmup steps
        use_scheduler (bool): whether or not to use a linearly decaying scheduler
    """

    lr: float = validated_field(1e-3, validator=LearningRate)
    """learning rate"""
    weight_decay: float = validated_field(0.0, validator=WeightDecay)
    """optimizer weight decay"""
    betas: tuple[float, float] = (0.9, 0.999)
    """optimizer betas"""
    warmup_steps: int = validated_field(0, validator=NonNegativeInt)
    """number of warmup steps"""
    use_scheduler: bool = False
    """whether or not to use a linearly decaying scheduler"""


LeftClosedRightOpenUnitInterval = restricted_number_type(
    "LeftClosedRightOpenUnitInterval",
    float,
    [(">=", 0.0), ("<", 1.0)],
    docstring="float in [0.0, 1.0)",
)


@dataclass
class BaseModelConfig(DataclassValidatorMixin):
    """MODEL

    Base config for all PST models, genomic or protein. This can be used as is, but subclassing
    can be used to add additional parameters. Additionally, custom loss and augmentation
    configs can be passed to the respective fields when subclassing. The subfields in the loss
    field are passed as additional arguments to the `setup_objective` function.

    Attributes:
        in_dim (int): input dimension
        out_dim (int): output dimension, default is to use the input dimension
        num_heads (int): number of attention heads
        n_enc_layers (int): number of encoder layers
        embed_scale (int): scale factor for positional and strand embeddings
        dropout (float): dropout rate for individual weight parameters, [0.0, 1.0)
        layer_dropout (float): dropout rate for entire layers, [0.0, 1.0)
        proj_cat (bool): whether to project the concatenated pLM, positional, and strand embeddings
        max_proteins (int): maximum number of proteins in a scaffold
        compile (bool): whether to compile the model with `torch.compile`
        optimizer (OptimizerConfig): OPTIMIZER
        loss (BaseLossConfig): LOSS
        augmentation (BaseAugmentationConfig): AUGMENTATION
    """

    # the above is used for inherited docstrings

    # however the attribute docstrings are used for IDEs,
    # while jsonargparse can use either depending on the situation
    in_dim: int
    """input dimension"""

    out_dim: int = -1
    """output dimension, default is to use the input dimension"""

    num_heads: int = validated_field(4, validator=PositiveInt)
    """number of attention heads"""

    n_enc_layers: int = validated_field(5, validator=PositiveInt)
    """number of encoder layers"""

    embed_scale: int = validated_field(4, validator=PositiveInt)
    """scale factor for positional and strand embeddings"""

    dropout: float = validated_field(0.0, validator=LeftClosedRightOpenUnitInterval)
    """dropout rate for individual weight parameters, [0.0, 1.0)"""

    layer_dropout: float = validated_field(
        0.0, validator=LeftClosedRightOpenUnitInterval
    )
    """dropout rate for entire layers, [0.0, 1.0)"""

    proj_cat: bool = False
    """whether to project the concatenated pLM, positional, and strand embeddings"""

    max_proteins: int = validated_field(MAX_PROTEINS_PER_GENOME, validator=PositiveInt)
    """maximum number of proteins in a scaffikd"""

    compile: bool = False
    """whether to compile the model with `torch.compile`"""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """OPTIMIZER
    
    config to setup a `torch.optim.AdamW` optimizer
    """

    loss: BaseLossConfig = field(default_factory=BaseLossConfig)
    """LOSS"""

    augmentation: BaseAugmentationConfig = field(default_factory=BaseAugmentationConfig)
    """AUGMENTATION"""

    def _validate_dimensions(self, attr: Literal["in_dim", "out_dim"]):
        value = getattr(self, attr)
        if value != -1 and value <= 0:
            raise ValueError(f"{attr} must be > 0 if specified")

    def __post_init__(self):
        super().__post_init__()

        for attr in ["in_dim", "out_dim"]:
            self._validate_dimensions(attr)  # type: ignore

        if (self.in_dim % self.num_heads) != 0:
            raise ValueError("in_dim must be divisible by num_heads if provided")

    def _get_original_dims(self) -> tuple[int, int]:
        """
        Get the original embedding dimensions before concatenation of positional and strand
        embeddings.

        The added dimension is: `D = 2 * embed_scale`
        So, the formula for the dim expansion is: `new_in_dim = 2 * D + in_dim`
        Thus, the formula for undoing this is: `in_dim = (new_in_dim * embed_scale) / (2 + embed_scale)

        Args:
            config (BaseModelConfig): the model config

        Returns:
            tuple[int, int]: the original input and output dimensions
        """

        new_in_dim = self.in_dim
        embed_scale = self.embed_scale

        original_in_dim = (new_in_dim * embed_scale) // (2 + embed_scale)

        if self.in_dim == self.out_dim:
            original_out_dim = original_in_dim
        else:
            original_out_dim = self.out_dim

        return original_in_dim, original_out_dim

    def _undo_dim_expansion(self):
        """
        The original input dimensions are expanded by PST models due to concatenating position
        and strand embeddings. This may need to be undone in certain cases, such as updating
        a pre-existing model's config, which would require creating a new model.

        This method undoes the dimension expansion in place.
        """
        original_in_dim, original_out_dim = self._get_original_dims()

        self.in_dim = original_in_dim
        self.out_dim = original_out_dim


# TODO: this should be renamed to triplet_loss_config
@dataclass
class GenomeTripletLossModelConfig(BaseModelConfig):
    """MODEL

    Model config for PST models that use triplet loss and PointSwap augmentation.

    Attributes:
        loss (GenomeTripletLossConfig): LOSS
        augmentation (AugmentationConfig): AUGMENTATION using PointSwap
    """

    loss: GenomeTripletLossConfig = field(default_factory=GenomeTripletLossConfig)
    """LOSS"""

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    """AUGMENTATION using PointSwap"""


MaskingRate = restricted_number_type(
    "MaskingRate",
    float,
    [(">=", 0.0), ("<", 0.5)],
    docstring="masking rate for MLM in [0.0, 0.5)",
)


@dataclass
class MaskedLanguageModelingLossConfig(WeightedLossConfig):
    """LOSS

    Loss config for masked language modeling.

    Attributes:
        masking_rate (float): masking rate for MLM in [0.0, 0.5).
    """

    masking_rate: float = validated_field(0.15, validator=MaskingRate)
    """masking rate for MLM"""


@dataclass
class MaskedLanguageModelingConfig(BaseModelConfig):
    """MODEL

    Model config for PSTs that use masked language modeling loss.

    Attributes:
        loss (MaskedLanguageModelingLossConfig): LOSS
    """

    loss: MaskedLanguageModelingLossConfig = field(
        default_factory=MaskedLanguageModelingLossConfig
    )
    """LOSS"""


UnionModelConfig = Union[
    BaseModelConfig, GenomeTripletLossModelConfig, MaskedLanguageModelingConfig
]

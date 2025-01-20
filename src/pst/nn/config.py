from typing import Union

from attrs import Attribute, define, field, validators

from pst.typing import NO_NEGATIVES_MODES
from pst.utils.attrs.dataclass_utils import AttrsDataclassUtilitiesMixin
from pst.utils.attrs.validators import (
    non_negative_int,
    open_unit_interval,
    optional_positive_int,
    positive_float,
    positive_int,
)

MAX_PROTEINS_PER_GENOME = 2048


@define
class BaseAugmentationConfig(AttrsDataclassUtilitiesMixin):
    """This is used to pass arguments to setting up the augmentation function.

    Subclass this if you need to pass additional arguments to the augmentation function.
    """

    pass


# TODO: rename to PointSwapAugmentationConfig
@define
class AugmentationConfig(BaseAugmentationConfig):
    """AUGMENTATION

    using PointSwap.

    Attributes:
        sample_rate (float): PointSwap sampler swapping rate in (0.0, 1.0)
    """

    sample_rate: float = field(default=0.5, validator=open_unit_interval)
    """PointSwap sampler swapping rate in (0.0, 1.0)"""


@define
class BaseLossConfig(AttrsDataclassUtilitiesMixin):
    """This is used to pass arguments to setting up the loss function.

    Subclass this if you need to pass additional arguments to the loss function.
    """

    pass


@define
class WeightedLossConfig(BaseLossConfig):
    """Base config for weighted loss functions that need a weight scale factor.

    Attributes:
        sample_scale (float): exponential decay scale factor for weighting samples during triplet, contrastive, or relational loss
    """

    sample_scale: float = field(default=7.0, validator=positive_float)
    """exponential decay scale factor for weighting samples during triplet, contrastive, or 
    relational loss"""


@define
class TripletLossConfig(WeightedLossConfig):
    """LOSS

    Triplet loss with exponential decay weighting.

    Attributes:
        margin (float): triplet loss margin
        no_negatives_mode (NO_NEGATIVES_MODES): mode to handle event of no semihard negative
            sample existing
    """

    margin: float = field(default=0.1, validator=positive_float)
    """triplet loss margin"""

    no_negatives_mode: NO_NEGATIVES_MODES = "closest_to_positive"
    """mode to handle event of no semihard negative sample existing"""


learning_rate_validator = validators.and_(
    validators.instance_of(float), validators.ge(1e-5), validators.le(1e-1)
)

weight_decay_validator = validators.and_(
    validators.instance_of(float), validators.ge(0.0), validators.le(1e-1)
)


@define
class OptimizerConfig(AttrsDataclassUtilitiesMixin):
    """OPTIMIZER
    for constructing a `torch.optim.AdamW` optimizer.

    Attributes:
        lr (float): learning rate
        weight_decay (float): optimizer weight decay
        betas (tuple[float, float]): optimizer betas
        warmup_steps (int): number of warmup steps
        use_scheduler (bool): whether or not to use a linearly decaying scheduler
    """

    lr: float = field(default=1e-3, validator=learning_rate_validator)
    """learning rate"""
    weight_decay: float = field(default=0.0, validator=weight_decay_validator)
    """optimizer weight decay"""
    betas: tuple[float, float] = (0.9, 0.999)
    """optimizer betas"""
    warmup_steps: int = field(default=0, validator=non_negative_int)
    """number of warmup steps"""
    use_scheduler: bool = False
    """whether or not to use a linearly decaying scheduler"""


left_closed_right_open_unit_interval = validators.and_(
    validators.instance_of(float), validators.ge(0.0), validators.lt(1.0)
)


def _in_dim_evenly_divisible_by_num_heads(
    instance: "BaseModelConfig", attribute: Attribute, value: int
):
    if (value % instance.num_heads) != 0:
        raise ValueError(
            f"in_dim must be divisible by num_heads. Received: in_dim={value} and "
            f"num_heads={instance.num_heads}"
        )


_in_dim_validator = validators.and_(positive_int, _in_dim_evenly_divisible_by_num_heads)


@define
class BaseModelConfig(AttrsDataclassUtilitiesMixin):
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
    in_dim: int = field(validator=_in_dim_validator)
    """input dimension"""

    out_dim: int = field(default=-1, validator=optional_positive_int)
    """output dimension, default is to use the input dimension"""

    num_heads: int = field(default=4, validator=positive_int)
    """number of attention heads"""

    n_enc_layers: int = field(default=5, validator=positive_int)
    """number of encoder layers"""

    embed_scale: int = field(
        default=4, validator=validators.and_(positive_int, validators.in_({1, 2, 4, 8}))
    )
    """scale factor for positional and strand embeddings"""

    dropout: float = field(default=0.0, validator=left_closed_right_open_unit_interval)
    """dropout rate for individual weight parameters, [0.0, 1.0)"""

    layer_dropout: float = field(
        default=0.0, validator=left_closed_right_open_unit_interval
    )
    """dropout rate for entire layers, [0.0, 1.0)"""

    proj_cat: bool = False
    """whether to project the concatenated pLM, positional, and strand embeddings"""

    max_proteins: int = field(default=MAX_PROTEINS_PER_GENOME, validator=positive_int)
    """maximum number of proteins in a scaffikd"""

    compile: bool = False
    """whether to compile the model with `torch.compile`"""

    optimizer: OptimizerConfig = field(factory=OptimizerConfig)
    """OPTIMIZER
    
    config to setup a `torch.optim.AdamW` optimizer
    """

    loss: BaseLossConfig = field(factory=BaseLossConfig)
    """LOSS"""

    # jsonargparse has problems when a sub attrs.define dataclass has no init args
    # augmentation: BaseAugmentationConfig = field(factory=BaseAugmentationConfig)
    # """AUGMENTATION"""

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

    # def __attrs_post_init__(self):
    #     # validate args with cattrs deserialization
    #     ser = self.to_dict()

    #     deser = self.__class__.from_dict(ser)

    #     if deser != self:
    #         raise ValueError(
    #             f"Failed to validate arguments. Constructed: {self}\nExpected: {deser}"
    #         )


@define
class ProteinTripletLossModelConfig(BaseModelConfig):
    """MODEL

    Model config for protein PST models that use triplet loss.

    Attributes:
        loss (TripletLossConfig): LOSS
    """

    loss: TripletLossConfig = field(factory=TripletLossConfig)
    """LOSS"""


@define
class GenomeTripletLossModelConfig(ProteinTripletLossModelConfig):
    """MODEL

    Model config for PST models that use triplet loss and PointSwap augmentation.

    Attributes:
        augmentation (AugmentationConfig): AUGMENTATION using PointSwap
    """

    augmentation: AugmentationConfig = field(factory=AugmentationConfig)
    """AUGMENTATION using PointSwap"""


masking_rate = validators.and_(
    validators.instance_of(float), validators.ge(0.0), validators.lt(0.5)
)


@define
class MaskedLanguageModelingLossConfig(WeightedLossConfig):
    """LOSS

    Loss config for masked language modeling.

    Attributes:
        masking_rate (float): masking rate for MLM in [0.0, 0.5).
    """

    masking_rate: float = field(default=0.15, validator=masking_rate)
    """masking rate for MLM"""


@define
class MaskedLanguageModelingConfig(BaseModelConfig):
    """MODEL

    Model config for PSTs that use masked language modeling loss.

    Attributes:
        loss (MaskedLanguageModelingLossConfig): LOSS
    """

    loss: MaskedLanguageModelingLossConfig = field(
        factory=MaskedLanguageModelingLossConfig
    )
    """LOSS"""


UnionModelConfig = Union[
    ProteinTripletLossModelConfig,
    GenomeTripletLossModelConfig,
    MaskedLanguageModelingConfig,
]

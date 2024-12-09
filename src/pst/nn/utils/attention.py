import math

from einops import rearrange
from torch import Tensor
from torch.nn import Linear


class AttentionMixin:
    def __init__(self, in_dim: int, heads: int):
        if in_dim % heads != 0:
            raise ValueError(f"dim {in_dim} not divisible by heads {heads}")

        self.heads = heads
        self.dim_per_head = in_dim // heads

    @property
    def attention_scale(self):
        return math.sqrt(self.dim_per_head)

    def uncat_heads(self, x: Tensor) -> Tensor:
        return rearrange(x, "... (heads dim) -> ... heads dim", heads=self.heads)

    def cat_heads(self, x: Tensor) -> Tensor:
        return rearrange(x, "... heads dim -> ... (heads dim)", heads=self.heads)

    def init_weight_layers(self, feature_dim: int) -> tuple[Linear, Linear, Linear]:
        linQ = Linear(feature_dim, feature_dim)
        linK = Linear(feature_dim, feature_dim)
        linV = Linear(feature_dim, feature_dim)
        return linQ, linK, linV

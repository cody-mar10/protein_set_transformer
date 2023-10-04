import math

from einops import rearrange
from torch import Tensor


def attention_scale(dim: int) -> float:
    return math.sqrt(dim)


def uncat_heads(x: Tensor, heads: int) -> Tensor:
    return rearrange(x, "... (heads dim) -> ... heads dim", heads=heads)


def cat_heads(x: Tensor) -> Tensor:
    return rearrange(x, "... heads dim -> ... (heads dim)")

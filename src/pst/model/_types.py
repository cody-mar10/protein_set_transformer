from __future__ import annotations

from torch import Tensor

AttentionOutput = tuple[Tensor, tuple[Tensor, Tensor]]
OptionalAttentionOutput = Tensor | AttentionOutput

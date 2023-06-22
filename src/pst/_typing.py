from __future__ import annotations

from typing import Literal, Protocol

from torch import Tensor
from torch_geometric.typing import OptTensor


# data.py types
class DataBatch(Protocol):
    x: Tensor
    y: OptTensor
    edge_index: Tensor
    batch: Tensor
    ptr: Tensor
    setsize: Tensor
    weight: Tensor
    class_id: Tensor


EdgeIndexStrategy = Literal["full", "sparse", "chunked"]

# layer.py types
AttentionOutput = tuple[Tensor, tuple[Tensor, Tensor]]
OptionalAttentionOutput = Tensor | AttentionOutput

# distance.py types
FlowType = dict[tuple[int, int], Tensor]

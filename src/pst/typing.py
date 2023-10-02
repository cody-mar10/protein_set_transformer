from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from torch import Tensor
from torch_geometric.typing import OptTensor, PairTensor

FilePath = str | Path


# data.py types
class GenomeGraphBatch(Protocol):
    x: Tensor
    y: OptTensor
    edge_index: Tensor
    batch: Tensor
    ptr: Tensor
    num_proteins: Tensor
    weight: Tensor
    class_id: Tensor
    strand: Tensor


EdgeIndexStrategy = Literal["full", "sparse", "chunked"]

# layer.py types
EdgeAttnOutput = tuple[Tensor, tuple[Tensor, Tensor]]
OptEdgeAttnOutput = Tensor | EdgeAttnOutput

GraphAttnOutput = PairTensor
OptGraphAttnOutput = Tensor | GraphAttnOutput

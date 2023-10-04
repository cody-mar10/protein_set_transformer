from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from torch import Tensor
from torch_geometric.typing import PairTensor  # noqa
from torch_geometric.typing import OptTensor

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
    pos: Tensor


EdgeIndexStrategy = Literal["full", "sparse", "chunked"]

# layer.py types
EdgeAttnOutput = tuple[Tensor, tuple[Tensor, Tensor]]
OptEdgeAttnOutput = Tensor | EdgeAttnOutput

OptGraphAttnOutput = tuple[Tensor, OptTensor]

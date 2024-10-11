from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple, Protocol

from numpy import float32
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.typing import (
    OptTensor,
    PairTensor,  # noqa
)

FilePath = str | Path

NO_NEGATIVES_MODES = Literal["closest_to_positive", "closest_to_anchor"]


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
    scaffold_label: Tensor
    genome_label: Tensor


EdgeIndexStrategy = Literal["full", "sparse", "chunked"]


# layer.py types
class EdgeAttnOutput(NamedTuple):
    out: Tensor
    edge_index: OptTensor
    attn: OptTensor


class GraphAttnOutput(NamedTuple):
    out: Tensor
    attn: OptTensor


NumpyFloat32 = NDArray[float32]

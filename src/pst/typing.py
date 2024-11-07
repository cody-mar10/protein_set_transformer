from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple, Optional, Protocol

from numpy import float32
from numpy.typing import NDArray
from torch import Tensor

FilePath = str | Path

NO_NEGATIVES_MODES = Literal["closest_to_positive", "closest_to_anchor"]
PairTensor = tuple[Tensor, Tensor]
OptTensor = Optional[Tensor]


# data.py types
class GenomeGraphBatch(Protocol):
    x: Tensor
    y: OptTensor
    edge_index: Tensor
    batch: Tensor
    ptr: Tensor
    num_proteins: Tensor
    strand: Tensor
    pos: Tensor
    scaffold_label: Tensor
    genome_label: Tensor


class MaskedGenomeGraphBatch(GenomeGraphBatch):
    node_mask: Tensor
    masked_embeddings: Tensor


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

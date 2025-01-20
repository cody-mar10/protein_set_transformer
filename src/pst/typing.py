from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, Protocol, TypeVar, Union

from numpy import float32
from numpy.typing import NDArray
from torch import Tensor

FilePath = Union[str, Path]

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


class EdgeIndexStrategy(str, Enum):
    full = "full"
    sparse = "sparse"
    chunked = "chunked"


# layer.py types
class EdgeAttnOutput(NamedTuple):
    out: Tensor
    edge_index: OptTensor
    attn: OptTensor


class GraphAttnOutput(NamedTuple):
    out: Tensor
    attn: OptTensor


NumpyFloat32 = NDArray[float32]

KwargType = dict[str, Any]
_T = TypeVar("_T")

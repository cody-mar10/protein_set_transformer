from __future__ import annotations

from typing import TypeVar

import tables as tb
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from pst.typing import EdgeIndexStrategy, FilePath, GenomeGraphBatch

from .graph import (
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_EDGE_STRATEGY,
    _SENTINEL_THRESHOLD,
    GenomeGraph,
)

GraphT = TypeVar("GraphT", bound=Data)


class GenomeDataset(Dataset[GenomeGraph]):
    __h5_fields__ = {"data", "ptr", "sizes", "class_id", "strand"}
    __node_attr__ = {"x", "strand"}
    __graph_attr__ = {"edge_index", "size", "weight", "class_id"}

    data: torch.Tensor
    ptr: torch.Tensor
    sizes: torch.Tensor
    class_id: torch.Tensor
    strand: torch.Tensor
    weights: torch.Tensor
    edge_indices: list[torch.Tensor]

    def __init__(
        self,
        file: FilePath,
        edge_strategy: EdgeIndexStrategy = _DEFAULT_EDGE_STRATEGY,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        threshold: int = _SENTINEL_THRESHOLD,
        log_inverse: bool = True,
    ) -> None:
        super().__init__()
        with tb.File(file) as fp:
            for field in GenomeDataset.__h5_fields__:
                data = getattr(fp.root, field)
                setattr(self, field, torch.from_numpy(data[:]))

        # convert strand array from [-1, 1] -> [0, 1]
        # this will be used as an idx in a lut embedding
        self.strand[self.strand == -1] = 0

        self.weights = self._get_class_weights(log_inverse)

        self.edge_indices = self._get_edge_indices(edge_strategy, chunk_size, threshold)

    def _get_class_weights(self, log_inverse: bool = True) -> torch.Tensor:
        if hasattr(self, "class_id"):
            # calc using inverse frequency
            # convert to ascending 0..n range
            class_counts: torch.Tensor
            _, class_counts = torch.unique(self.class_id, return_counts=True)
            freq: torch.Tensor = class_counts / class_counts.sum()
            inv_freq = 1.0 / freq
            if log_inverse:
                # with major class imbalance the contribution from rare classes can
                # be extremely high relative to other classes
                inv_freq = torch.log(inv_freq)

            # not sure if normalization does anything since all still contribute
            # the relative same amount to loss
            inv_freq /= torch.amin(inv_freq)
            weights = inv_freq[self.class_id]
        else:
            # no weights
            weights = torch.ones(size=(len(self),))

        return weights

    def _get_edge_indices(
        self,
        edge_strategy: EdgeIndexStrategy,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        threshold: int = _SENTINEL_THRESHOLD,
    ) -> list[torch.Tensor]:
        edge_indices: list[torch.Tensor] = list()
        edge_create_fn = GenomeGraph._edge_index_create_method(
            edge_strategy=edge_strategy, chunk_size=chunk_size, threshold=threshold
        )

        for num_nodes in self.sizes:
            edge_index = edge_create_fn(num_nodes=num_nodes)
            edge_indices.append(edge_index)

        return edge_indices

    def __len__(self) -> int:
        return self.sizes.numel()

    def __getitem__(self, idx: int) -> GenomeGraph:
        # idx should be for a single genome
        try:
            start = self.ptr[idx]
            stop = self.ptr[idx + 1]
        except IndexError as e:
            clsname = self.__class__.__name__
            raise IndexError(
                f"{idx=} is out of range for {clsname} with {len(self)} genomes"
            ) from e

        # node/item level access (ptn)
        x = self.data[start:stop]
        strand = self.strand[start:stop]

        # graph/set level access (genome)
        edge_index = self.edge_indices[idx]
        num_proteins = self.sizes[idx]
        weight = self.weights[idx]
        class_id = self.class_id[idx]

        # the edge index will already be created so no need to pass edge creation info
        graph = GenomeGraph(
            x=x,
            edge_index=edge_index,
            num_proteins=int(num_proteins),
            weight=weight.item(),
            class_id=int(class_id),
            strand=strand,
        )
        return graph

    @property
    def feature_dim(self) -> int:
        return int(self.data.shape[-1])

    @property
    def max_size(self) -> int:
        return int(self.sizes.amax())

    @staticmethod
    def collate(batch: list[GraphT]) -> GenomeGraphBatch:
        return Batch.from_data_list(batch)  # type: ignore

    def collate_indices(self, idx_batch: list[torch.Tensor]) -> GenomeGraphBatch:
        batch = [self[int(idx)] for idx in idx_batch]
        return self.collate(batch)

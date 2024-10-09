from __future__ import annotations

from typing import Literal, TypeVar

import tables as tb
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from pst.data.graph import (
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_EDGE_STRATEGY,
    _SENTINEL_THRESHOLD,
    GenomeGraph,
)
from pst.typing import EdgeIndexStrategy, FilePath, GenomeGraphBatch

GraphT = TypeVar("GraphT", bound=Data)
FeatureLevel = Literal["node", "graph", "protein", "genome"]


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
                try:
                    data = getattr(fp.root, field)
                except tb.exceptions.NoSuchNodeError:
                    if field == "class_id":
                        # the class_id field is not required for inference
                        # this was only used for weighting the loss
                        continue
                    else:
                        raise
                setattr(self, field, torch.from_numpy(data[:]))

        # convert strand array from [-1, 1] -> [0, 1]
        # this will be used as an idx in a lut embedding
        self.strand[self.strand == -1] = 0

        self.weights = self._get_class_weights(log_inverse)

        if not hasattr(self, "class_id"):
            # default all genomes to the same class
            self.class_id = torch.zeros((len(self),), dtype=torch.long).numpy()

        self.edge_indices = self._get_edge_indices(edge_strategy, chunk_size, threshold)

        self._node_registry: dict[str, torch.Tensor] = dict()
        self._graph_registry: dict[str, torch.Tensor] = dict()

    def _get_class_weights(self, log_inverse: bool = True) -> torch.Tensor:
        if hasattr(self, "class_id"):
            # calc using inverse frequency
            # convert to ascending 0..n range
            class_counts: torch.Tensor
            _, inverse_index, class_counts = torch.unique(
                self.class_id, return_inverse=True, return_counts=True
            )
            freq: torch.Tensor = class_counts / class_counts.sum()
            inv_freq = 1.0 / freq
            if log_inverse:
                # with major class imbalance the contribution from rare classes can
                # be extremely high relative to other classes
                inv_freq = torch.log(inv_freq)

            # not sure if normalization does anything since all still contribute
            # the relative same amount to loss
            inv_freq /= torch.amin(inv_freq)

            # inverse index remaps input class_ids to 0..n range if not already
            weights = inv_freq[inverse_index]
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

        registered_features: dict[str, torch.Tensor | int | float] = {}

        # node/item level access (ptn)
        x = self.data[start:stop]
        strand = self.strand[start:stop]

        for name, data in self._node_registry.items():
            registered_features[name] = data[start:stop]

        # graph/set level access (genome)
        edge_index = self.edge_indices[idx]
        num_proteins = int(self.sizes[idx])
        weight = self.weights[idx].item()
        class_id = int(self.class_id[idx])
        # shape: [N, 1]
        pos = torch.arange(num_proteins).unsqueeze(-1).to(x.device)

        for name, data in self._graph_registry.items():
            registered_features[name] = data[idx]

        # the edge index will already be created so no need to pass edge creation info
        graph = GenomeGraph(
            x=x,
            edge_index=edge_index,
            num_proteins=num_proteins,
            weight=weight,
            class_id=class_id,
            strand=strand,
            pos=pos,
            **registered_features,  # type: ignore
        )
        return graph

    def register_feature(
        self,
        name: str,
        data: torch.Tensor,
        *,
        feature_level: FeatureLevel,
        overwrite_previously_registered: bool = False,
    ):
        if name in self.__node_attr__ or name in self.__graph_attr__:
            raise ValueError(
                f"Cannot register feature with name {name} as it is a reserved attribute."
            )
        elif name in self._node_registry or name in self._graph_registry:
            if not overwrite_previously_registered:
                raise ValueError(
                    f"Feature with name {name} has already been registered. "
                    "Set `overwrite_previously_registered=True` to overwrite."
                )
            else:
                self._node_registry.pop(name, None)
                self._graph_registry.pop(name, None)

        if feature_level == "node" or feature_level == "protein":
            if data.shape[0] != self.data.shape[0]:
                raise ValueError(
                    f"Expected {self.data.shape[0]} items in data, got {data.shape[0]}"
                )
            self._node_registry[name] = data
        elif feature_level == "graph" or feature_level == "genome":
            if data.shape[0] != self.sizes.shape[0]:
                raise ValueError(
                    f"Expected {self.sizes.shape[0]} items in data, got {data.shape[0]}"
                )
            self._graph_registry[name] = data
        else:
            raise ValueError(
                (
                    "Invalid feature level. Must be one of 'node'/'protein' or 'graph'/'genome'"
                    "to indicate if the registered feature describes a node-level or "
                    "graph-level property, respectively."
                )
            )

    @property
    def feature_dim(self) -> int:
        return int(self.data.shape[-1])

    @property
    def max_size(self) -> int:
        return int(self.sizes.amax())

    @staticmethod
    def collate(batch: list[GenomeGraph]) -> GenomeGraphBatch:
        return Batch.from_data_list(batch)  # type: ignore

    def collate_indices(self, idx_batch: list[torch.Tensor]) -> GenomeGraphBatch:
        batch = [self[int(idx)] for idx in idx_batch]
        return self.collate(batch)

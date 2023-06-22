from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Iterator, Literal

import lightning as L
import tables as tb
import torch
from more_itertools import chunked
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from ._types import DataBatch, EdgeIndexStrategy
from pst.training.cross_validation import ImbalancedGroupKFold

FilePath = str | Path
_DEFAULT_CHUNK_SIZE = 30
_SENTINEL_THRESHOLD = -1
_DEFAULT_THRESHOLD = 30


def create_fully_connected_graph(num_nodes: int) -> torch.Tensor:
    edge_index = (
        torch.tensor(list(permutations(range(num_nodes), r=2))).t().contiguous()
    )
    return edge_index


def filter_edges_by_seq_distance(
    edge_index: torch.Tensor, threshold: int
) -> torch.Tensor:
    distance = torch.abs(edge_index[0] - edge_index[1])
    local_edge_index = edge_index[:, distance <= threshold].contiguous()
    return local_edge_index


def create_sparse_graph(num_nodes: int, threshold: int) -> torch.Tensor:
    edge_index = create_fully_connected_graph(num_nodes)
    edge_index = filter_edges_by_seq_distance(edge_index, threshold)
    return edge_index


def create_chunked_graph(
    num_nodes: int, chunk_size: int, threshold: int = _SENTINEL_THRESHOLD
) -> torch.Tensor:
    connected_comp = list(chunked(range(num_nodes), n=chunk_size))
    # don't want any connected components / subgraphs that only have 1 node
    if len(connected_comp[-1]) == 1:
        connected_comp[-2].append(connected_comp[-1][0])
        del connected_comp[-1]

    # False if threshold == -1 or >= chunk_size
    # True if threshold < chunk_size
    filter_edges = not (threshold == _SENTINEL_THRESHOLD or threshold >= chunk_size)

    _edge_index: list[torch.Tensor] = list()
    offset = 0
    for cc in connected_comp:
        cc_num_nodes = len(cc)
        edges = create_fully_connected_graph(cc_num_nodes) + offset
        if filter_edges:
            edges = filter_edges_by_seq_distance(edges, threshold)
        _edge_index.append(edges)
        offset += cc_num_nodes

    edge_index = torch.cat(_edge_index, dim=1)
    return edge_index


class GenomeDataset(Dataset):
    def __init__(
        self,
        file: FilePath,
        edge_strategy: EdgeIndexStrategy,
        **kwargs,
    ) -> None:
        super().__init__()
        with tb.File(file) as fp:
            self.data: torch.Tensor = torch.from_numpy(fp.root.data[:])
            self.ptr: torch.Tensor = torch.from_numpy(fp.root.ptr[:])
            self.sizes: torch.Tensor = torch.from_numpy(fp.root.sizes[:])
            # TODO: may want to calculate class weights per training fold
            # if you do simple normalized inverse freq, then the values are the same across all folds
            # and then you could just calculate this once
            # but if you first take the log of inverse freq like these weights are,
            # then values differ for the same class per fold, which may introduce some variability
            # for now, will just leave this alone
            self.weights: torch.Tensor = torch.from_numpy(fp.root.weights[:])
            self.class_id: torch.Tensor = torch.from_numpy(fp.root.class_id[:])

        self.edge_indices = self.get_edge_indices(edge_strategy, **kwargs)

    def get_edge_indices(
        self, edge_strategy: EdgeIndexStrategy, **kwargs
    ) -> list[torch.Tensor]:
        edge_indices: list[torch.Tensor] = list()
        if edge_strategy == "sparse":
            if "threshold" not in kwargs:
                raise ValueError(
                    f"Passed {edge_strategy=}, which requires the `threshold` arg for `create_sparse_graph`"
                )

            edge_create_fn = create_sparse_graph
        elif edge_strategy == "chunked":
            if "chunk_size" not in kwargs:
                raise ValueError(
                    f"Passed {edge_strategy=}, which requires the `chunk_size` arg for `create_chunked_graph`"
                )

            edge_create_fn = create_chunked_graph
        else:
            kwargs = dict()
            edge_create_fn = create_fully_connected_graph

        for num_nodes in self.sizes:
            edge_index = edge_create_fn(num_nodes=num_nodes, **kwargs)  # type: ignore
            edge_indices.append(edge_index)

        return edge_indices

    def __len__(self) -> int:
        return self.sizes.numel()

    def __getitem__(self, idx: int) -> Data:
        start = self.ptr[idx]
        stop = self.ptr[idx + 1]
        x = self.data[start:stop]
        edge_index = self.edge_indices[idx]
        size = self.sizes[idx]
        weight = self.weights[idx]
        class_id = self.class_id[idx]
        graph = Data(
            x=x,
            edge_index=edge_index,
            setsize=size,
            weight=weight,
            class_id=class_id,
        )
        return graph

    @staticmethod
    def collate(batch: list[Data]) -> DataBatch:
        return Batch.from_data_list(batch)  # type: ignore


_StageType = Literal["fit", "test", "predict"]


class SimpleTensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.data = tensor

    def __len__(self) -> int:
        return int(self.data.size(0))

    def __getitem__(self, idx: int | list[int]) -> torch.Tensor:
        return self.data[idx]

    @staticmethod
    def collate(batch: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch)


def indices_to_genome_data_batch(
    idx_batch: list[torch.Tensor], genome_dataset: GenomeDataset
) -> DataBatch:
    batch = [genome_dataset[int(idx)] for idx in idx_batch]
    return GenomeDataset.collate(batch)


class GenomeDataModule(L.LightningDataModule):
    def __init__(
        self,
        file: FilePath,
        batch_size: int,
        edge_strategy: EdgeIndexStrategy = "chunked",
        train_on_full: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        edge_strat_kwargs = self._prune_init_kwargs_for_edge_strategy(
            edge_strategy, **kwargs
        )
        self.dataloader_kwargs = kwargs
        self.dataset = GenomeDataset(
            file=file, edge_strategy=edge_strategy, **edge_strat_kwargs
        )
        self.batch_size = batch_size
        self.train_on_full = train_on_full
        self.save_hyperparameters()

    @staticmethod
    def _prune_init_kwargs_for_edge_strategy(
        edge_strategy: EdgeIndexStrategy, **kwargs
    ) -> dict[str, int]:
        edge_strat_kwargs: dict[str, int] = dict()
        if edge_strategy == "chunked":
            defaults = {
                "chunk_size": _DEFAULT_CHUNK_SIZE,
                "threshold": _SENTINEL_THRESHOLD,
            }
        elif edge_strategy == "sparse":
            defaults = {
                "threshold": _DEFAULT_THRESHOLD,
            }
        else:
            defaults = dict()

        for key, default in defaults.items():
            edge_strat_kwargs[key] = kwargs.pop(key, default)

        return edge_strat_kwargs

    def _convert_data_indices_to_genome_data_batch(
        self, idx_batch: list[torch.Tensor]
    ) -> DataBatch:
        return indices_to_genome_data_batch(
            idx_batch=idx_batch, genome_dataset=self.dataset
        )

    def setup(self, stage: _StageType):
        if stage == "fit":
            if self.train_on_full:
                # train final model with all data
                self.train_dataset = self.dataset
            else:
                # train with cross validation
                self.data_manager = ImbalancedGroupKFold(groups=self.dataset.class_id)
        elif stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def train_val_dataloaders(
        self, **kwargs
    ) -> Iterator[tuple[DataLoader, DataLoader]]:
        # pseudocode:
        train_idx: torch.Tensor
        val_idx: torch.Tensor
        for train_idx, val_idx in self.data_manager.split(return_torch=True):  # type: ignore
            train_idx_dataset = SimpleTensorDataset(train_idx)
            val_idx_dataset = SimpleTensorDataset(val_idx)
            train_loader = DataLoader(
                dataset=train_idx_dataset,
                batch_size=self.batch_size,
                collate_fn=self._convert_data_indices_to_genome_data_batch,
                **kwargs,
            )
            val_loader = DataLoader(
                dataset=val_idx_dataset,
                batch_size=self.batch_size,
                collate_fn=self._convert_data_indices_to_genome_data_batch,
                **kwargs,
            )
            yield train_loader, val_loader

    def simple_dataloader(self, dataset: GenomeDataset) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            **self.dataloader_kwargs,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.simple_dataloader(self.train_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.simple_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self.simple_dataloader(self.predict_dataset)

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Literal

import tables as tb
import torch
from lightning_cv import CrossValidationDataModule
from lightning_cv.split import ImbalancedLeaveOneGroupOut
from more_itertools import chunked
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from pst._typing import DataBatch, EdgeIndexStrategy

FilePath = str | Path
_DEFAULT_CHUNK_SIZE = 30
_SENTINEL_THRESHOLD = -1
_DEFAULT_THRESHOLD = 30


class DataConfig(BaseModel):
    file: Path = Field(
        ...,
        description="input protein embeddings file in .h5 file format with the fields .data for protein embeddings",  # TODO: add more desc  # noqa: E501
    )
    batch_size: int = Field(
        32, description="batch size in number of genomes", gt=4, le=128, multiple_of=2
    )
    train_on_full: bool = Field(
        False,
        description=(
            "whether to train a single model on the full input data. When not "
            "specified, the default is  to train multiple models with cross "
            "validation"
        ),
    )
    pin_memory: bool = Field(
        True, description="whether to pin memory onto a CUDA GPU or not"
    )
    num_workers: int = Field(0, description="additional cpu workers to load data", ge=0)
    edge_strategy: EdgeIndexStrategy = Field(
        "chunked",
        description=(
            "strategy to create 'edges' between protein nodes in a genome graph. "
            "chunked = split genomes in --chunk-size chunks. sparse = remove "
            "interactions longer than --threshold. full = fully connected graph "
            "like a regular transformer."
        ),
    )
    chunk_size: int = Field(
        30,
        description="size of sub-chunks to break genomes into if using --edge_strategy chunked",  # noqa: E501
        ge=15,
        le=50,
    )
    threshold: int = Field(
        -1,
        description="range of protein interactions if using --edge_strategy [chunked|sparse]",  # noqa: E501,
    )
    log_inverse: bool = Field(
        False, description="take the log of inverse class freqs as weights"
    )


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
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        threshold: int = _SENTINEL_THRESHOLD,
        log_inverse: bool = True,
    ) -> None:
        super().__init__()
        with tb.File(file) as fp:
            self.data: torch.Tensor = torch.from_numpy(fp.root.data[:])
            self.ptr: torch.Tensor = torch.from_numpy(fp.root.ptr[:])
            self.sizes: torch.Tensor = torch.from_numpy(fp.root.sizes[:])
            self.class_id: torch.Tensor = torch.from_numpy(fp.root.class_id[:])

        self.weights = self.get_class_weights(log_inverse)

        self.edge_indices = self.get_edge_indices(edge_strategy, chunk_size, threshold)

    def get_class_weights(self, log_inverse: bool = True) -> torch.Tensor:
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

    def get_edge_indices(
        self,
        edge_strategy: EdgeIndexStrategy,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        threshold: int = _SENTINEL_THRESHOLD,
    ) -> list[torch.Tensor]:
        edge_indices: list[torch.Tensor] = list()
        kwargs = dict()
        if edge_strategy == "sparse":
            if threshold <= 1:
                errmsg = (
                    f"Passed {edge_strategy=}, which requires the `threshold`"
                    " arg for `create_sparse_graph` to be >1"
                )
                raise ValueError(errmsg)
            kwargs["threshold"] = threshold
            edge_create_fn = create_sparse_graph
        elif edge_strategy == "chunked":
            kwargs["threshold"] = threshold
            kwargs["chunk_size"] = chunk_size
            edge_create_fn = create_chunked_graph
        else:
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

    @property
    def feature_dim(self) -> int:
        return int(self.data.shape[-1])

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


@dataclass
class CVDataLoader:
    train_loader: DataLoader
    val_loader: DataLoader
    train_group_ids: list[int]
    val_group_id: int


_KwargType = dict[str, Any]


class GenomeDataModule(CrossValidationDataModule):
    _LOGGABLE_HPARAMS = {
        "batch_size",
        "edge_strategy",
        "chunk_size",
        "threshold",
        "log_inverse",
        "train_on_full",
    }

    def __init__(self, config: DataConfig, **kwargs) -> None:
        self.config = config
        self.dataset = GenomeDataset(
            file=config.file,
            edge_strategy=config.edge_strategy,
            chunk_size=config.chunk_size,
            threshold=config.threshold,
            log_inverse=config.log_inverse,
        )
        super().__init__(
            dataset=self.dataset,
            batch_size=config.batch_size,
            cross_validator=ImbalancedLeaveOneGroupOut,
            cross_validator_config={"groups": self.dataset.class_id},
            collate_fn=self._convert_data_indices_to_genome_data_batch,
            **kwargs,
        )

        # shared / global dataloader kwargs, but they can still be updated
        self.dataloader_kwargs = kwargs

        self.save_hyperparameters(config.dict(include=self._LOGGABLE_HPARAMS))

    # collate_fn
    def _convert_data_indices_to_genome_data_batch(
        self, idx_batch: list[torch.Tensor]
    ) -> DataBatch:
        return indices_to_genome_data_batch(
            idx_batch=idx_batch, genome_dataset=self.dataset
        )

    def setup(self, stage: _StageType):
        if stage == "fit":
            if self.config.train_on_full:
                # train final model with all data
                self.train_dataset = self.dataset
            else:
                # train with cross validation
                # sets self.data_manager module which keeps track of the folds
                super().setup("fit")
        elif stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def simple_dataloader(self, dataset: GenomeDataset, **kwargs) -> DataLoader:
        kwargs = self._overwrite_dataloader_kwargs(**kwargs)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            **kwargs,
        )
        return dataloader

    def train_dataloader(self, **kwargs) -> DataLoader:
        return self.simple_dataloader(self.train_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return self.simple_dataloader(self.test_dataset, **kwargs)

    def predict_dataloader(self, **kwargs) -> DataLoader:
        return self.simple_dataloader(self.predict_dataset, **kwargs)

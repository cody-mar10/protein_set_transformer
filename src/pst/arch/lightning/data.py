from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import lightning as L
import tables as tb
import torch

# from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)

from pst.utils._types import BatchType

BATCH_SIZE = 128


def class_imbalance_sampler(
    labels: torch.Tensor, with_replacement: bool = False
) -> WeightedRandomSampler:
    """Create a PyTorch sampler for dataset with class imbalance.

    Args:
        labels (torch.Tensor): dummy encoded labels for classes
        with_replacement (bool, optional): whether to sample with replacement or not.
            This will help ensure all mini-batches have relatively homogeneous
            distributions but will repeat the rarer classes. Defaults to False.

    Returns:
        WeightedRandomSampler: PyTorch sampler obj to pass to a PyTorch DataLoader
    """
    class_counts = torch.bincount(labels)
    # simple inverse freq leads to a burnout effect where initial batches have each class
    # but then later classes tend to just be the most common class
    # using log of inv leads a much more uniform sampling regime
    class_sampling_weights = torch.log(1.0 / class_counts)
    class_sampling_weights /= class_sampling_weights.sum()
    full_sample_weights = class_sampling_weights[labels]
    sampler = WeightedRandomSampler(
        weights=full_sample_weights,  # type: ignore
        num_samples=labels.size(0),
        replacement=with_replacement,
    )
    return sampler


@dataclass
class SimpleMetadata:
    genome: str
    taxonomy: str
    taxonomy_weight: float
    slice: slice
    label: str
    label_id: int


class GenomeDataset(Dataset):
    def __init__(self, data_file: Path, genome_metadata: Path) -> None:
        self._data = torch.from_numpy(
            tb.File(data_file, libver="latest").root.data[:]
        ).cpu()
        self._genome_metadata = self.read_metadata(genome_metadata)

    def read_metadata(self, file: Path) -> list[SimpleMetadata]:
        with file.open() as fp:
            metadata: list[SimpleMetadata] = list()
            class_id_map: dict[str, int] = dict()

            start = 0
            class_id = 0
            for line in islice(fp, 1, None):
                (
                    genome,
                    _,
                    _nptns,
                    taxonomy,
                    _taxonomy_weight,
                    label,
                ) = line.rstrip().split("\t")
                nptns = int(_nptns)
                taxonomy_weight = float(_taxonomy_weight)
                end = start + nptns
                if label not in class_id_map:
                    class_id_map[label] = class_id
                    class_id += 1

                label_id = class_id_map[label]
                metadata.append(
                    SimpleMetadata(
                        genome=genome,
                        taxonomy=taxonomy,
                        taxonomy_weight=taxonomy_weight,
                        slice=slice(start, end),
                        label=label,
                        label_id=label_id,
                    )
                )
                start = end

        return metadata

    def get_class_labels(self) -> torch.Tensor:
        return torch.tensor([metadata.label_id for metadata in self._genome_metadata])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        # idx is the genome id
        genome_slice = self._genome_metadata[idx].slice
        return self._data[genome_slice], self._genome_metadata[idx].taxonomy_weight

    def __len__(self) -> int:
        return len(self._genome_metadata)

    def collate_batch(self, batch: list[tuple[torch.Tensor, float]]) -> BatchType:
        batch_data: list[torch.Tensor]
        batch_class_weights: list[float]
        batch_data, batch_class_weights = zip(*batch)  # type: ignore
        data = pad_sequence(batch_data, batch_first=True, padding_value=0.0)
        batch_taxonomy_weights = torch.Tensor(batch_class_weights)
        return data, batch_taxonomy_weights


class GenomeSetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_file: Path,
        metadata_file: Path,
        batch_size: int = BATCH_SIZE,
        split_ratio: tuple[float, float] = (0.8, 0.2),
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self._data_file = data_file
        self._metadata_file = metadata_file
        self.batch_size = batch_size
        self._dataset = GenomeDataset(self._data_file, self._metadata_file)
        self.loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)

        self.save_hyperparameters()

    @property
    def feature_dimension(self) -> int:
        return int(self._dataset._data.shape[-1])

    def setup(self, stage: str):
        # each process gets this

        if stage == "fit":
            self.train_dataset, self.val_dataset = random_split(
                self._dataset, lengths=self.hparams["split_ratio"]
            )
            class_labels = self._dataset.get_class_labels()
            train_class_labels = torch.tensor(
                [class_labels[train_idx] for train_idx in self.train_dataset.indices]
            )
            self.train_sampler = class_imbalance_sampler(train_class_labels)

        elif stage == "test":
            self.test_dataset = self._dataset
        elif stage == "predict":
            self.predict_dataset = self._dataset

    def _dataloader(
        self, dataset: GenomeDataset | Subset, stage: str, **kwargs
    ) -> DataLoader:
        if isinstance(dataset, Subset):
            # this only happens during the train/val loop due to random splitting
            if stage == "train":
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    sampler=self.train_sampler,
                    collate_fn=dataset.dataset.collate_batch,  # type: ignore
                    **kwargs,
                )
                return dataloader

            # val loop doesn't need a sampler
            assert stage == "val"
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.dataset.collate_batch,  # type: ignore
                **kwargs,
            )
            return dataloader

        # outside train/val loop
        assert isinstance(dataset, GenomeDataset) and stage not in {"train", "val"}
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_batch,
            **kwargs,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, stage="train", **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, stage="val", **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, stage="test", **self.loader_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.predict_dataset, stage="predict", **self.loader_kwargs
        )

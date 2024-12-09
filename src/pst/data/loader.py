from math import ceil
from typing import Iterator, cast

import torch
from torch.utils.data import DataLoader, Sampler, Subset

from pst.data.dataset import GenomeDataset, SubsetGenomeDataset
from pst.typing import GenomeGraphBatch

_ALLOWED_DATASET_TYPES = GenomeDataset | SubsetGenomeDataset | Subset[GenomeGraphBatch]
_COLLATE_FN = GenomeDataset.collate


class GenomeBatchSampler(Sampler[list[int]]):
    """Sampler to yield all scaffold-level indices for each genome in a `GenomeDataset`."""

    dataset: GenomeDataset

    def __init__(
        self,
        dataset: _ALLOWED_DATASET_TYPES,
        batch_size: int,
        shuffle: bool = False,
    ):
        self._dataset = dataset
        if not isinstance(dataset, GenomeDataset):
            # then it must be a subset
            self.dataset = dataset.dataset  # type: ignore
            self.num_genomes = len(dataset.indices)
            self.is_subset = True
        else:
            self.dataset = dataset
            self.num_genomes = dataset.num_genomes
            self.is_subset = False

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batches = (self.num_genomes + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            genome_indices = torch.randperm(self.num_genomes)
        else:
            genome_indices = torch.arange(self.num_genomes)

        genome_batch_indices: tuple[torch.Tensor, ...] = genome_indices.split(
            self.batch_size
        )

        if not self.is_subset:
            for genome_batch in genome_batch_indices:
                scaffold_idx = torch.where(
                    torch.isin(
                        self.dataset.scaffold_genome_label,
                        genome_batch,
                    )
                )[0]
                yield scaffold_idx.tolist()
        else:
            # if it is a subset, genome_batch_indices are relative to subset.indices
            self._dataset = cast(SubsetGenomeDataset, self._dataset)
            for genome_batch in genome_batch_indices:
                genome_idx = torch.tensor(
                    [self._dataset.indices[int(x)] for x in genome_batch]
                )

                scaffold_idx = torch.where(
                    torch.isin(
                        self.dataset.scaffold_genome_label,
                        genome_idx,
                    )
                )[0]

                yield scaffold_idx.tolist()

    def __len__(self):
        return self.num_batches


class GenomeDataLoader(DataLoader[GenomeGraphBatch]):
    """Load `GenomeGraphBatch` batches that include all scaffolds from each genome sampled.

    WARNING: batches will have dynamic shapes depending on the number of scaffolds in each
    genome and number of proteins encoded in each scaffold. It is recommended that the batch
    size not become too large to avoid memory issues.
    """

    dataset: _ALLOWED_DATASET_TYPES

    def __init__(
        self,
        dataset: _ALLOWED_DATASET_TYPES,
        batch_size: int,
        shuffle: bool = False,
        **kwargs,
    ):
        if "sampler" in kwargs:
            raise ValueError(
                "Cannot specify sampler for `GenomeDataLoader`. It is automatically set to `GenomeBatchSampler`."
            )

        self._original_dataset = dataset

        sampler = GenomeBatchSampler(dataset, batch_size, shuffle)

        if not isinstance(dataset, GenomeDataset):
            # doesn't work to pass the subset due to the way the sampler works, so just pass the original dataset
            dataset = dataset.dataset  # type: ignore

        super().__init__(
            dataset,
            batch_size=1,  # default
            sampler=None,
            shuffle=None,
            batch_sampler=sampler,
            collate_fn=_COLLATE_FN,
            **kwargs,
        )

    def __len__(self) -> int:
        if isinstance(self._original_dataset, GenomeDataset):
            num_genomes = self._original_dataset.num_genomes
        else:
            dataset = cast(GenomeDataset, self._original_dataset.dataset)
            indices = self._original_dataset.indices

            genome_labels = dataset.scaffold_genome_label[indices]
            num_genomes = int(torch.unique(genome_labels).numel())

        batch_size: int = self.batch_sampler.batch_size  # type: ignore
        if self.drop_last:
            length = num_genomes // batch_size
        else:
            length = ceil(num_genomes / batch_size)
        return length

    def __iter__(self) -> Iterator[GenomeGraphBatch]:
        return super().__iter__()


class ScaffoldDataLoader(DataLoader[GenomeGraphBatch]):
    dataset: _ALLOWED_DATASET_TYPES

    def __init__(
        self,
        dataset: _ALLOWED_DATASET_TYPES,
        batch_size: int,
        shuffle: bool = False,
        **kwargs,
    ):
        collate_fn = kwargs.pop("collate_fn", _COLLATE_FN)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

    def __iter__(self) -> Iterator[GenomeGraphBatch]:
        return super().__iter__()


class EmptyDataLoader(DataLoader):
    def __init__(self):
        super().__init__([], batch_size=1)  # type: ignore

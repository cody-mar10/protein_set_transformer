from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import DataLoader, Sampler

from pst.data.dataset import GenomeDataset
from pst.typing import GenomeGraphBatch


class GenomeBatchSampler(Sampler[list[int]]):
    """Sampler to yield all scaffold-level indices for each genome in a `GenomeDataset`."""

    def __init__(self, dataset: GenomeDataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.num_genomes = int(self.dataset.genome_label.max())
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batches = (self.num_genomes + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:

        if self.shuffle:
            genome_indices = torch.randperm(self.num_genomes)
        else:
            genome_indices = torch.arange(self.num_genomes)

        genome_batch_indices = genome_indices.split(self.batch_size)

        for genome_batch in genome_batch_indices:
            scaffold_idx = torch.where(
                torch.isin(
                    self.dataset.genome_label,
                    genome_batch,
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

    def __init__(
        self, dataset: GenomeDataset, batch_size: int, shuffle: bool = False, **kwargs
    ):
        if "sampler" in kwargs:
            raise ValueError(
                "Cannot specify sampler for `GenomeDataLoader`. It is automatically set to `GenomeBatchSampler`."
            )
        sampler = GenomeBatchSampler(dataset, batch_size, shuffle)
        super().__init__(
            dataset,
            batch_size=None,
            sampler=None,
            shuffle=None,
            batch_sampler=sampler,
            collate_fn=dataset.collate,
            **kwargs,
        )

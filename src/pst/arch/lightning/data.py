from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice, cycle
from pathlib import Path
from random import sample as random_sample
from typing import Iterable, Iterator, Optional, overload

import lightning as L
import tables as tb
import torch
from numpy import float32
from numpy.typing import NDArray

# from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler

BATCH_SIZE = 128


# TODO: genome2nptns and genome2idx could be merged into a single dict
@dataclass
class GenomeMetadata:
    id2genome: dict[int, str]
    genome2nptns: dict[str, int]
    genome2idx: dict[str, tuple[int, int]]

    def __len__(self) -> int:
        return len(self.id2genome)


def all_equal(data: Iterable) -> bool:
    iterator = iter(data)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(x == first for x in iterator)


_CACHE_SIZE = 10 * (1024**3)  # 10GB


class GenomeDataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        genome_metadata: Path,
        cache_size_bytes: int = _CACHE_SIZE,
    ) -> None:
        super(GenomeDataset, self).__init__()
        # chunkshape = tb.File(data_file).root.data.chunkshape
        # chunksize = chunkshape[0] * chunkshape[1] * 4
        # max_chunks_per_cache = cache_size_bytes / chunksize
        # nslots = round(100 * max_chunks_per_cache)
        self._filehandle = tb.File(
            data_file,
            rdcc_nbytes=cache_size_bytes,
            rdcc_nslots=821,
            libver="latest",
        )
        self._data = self._filehandle.root.data
        self._genome_metadata = self.read_metadata(genome_metadata)

    def __len__(self) -> int:
        # how many genomes
        return len(self._genome_metadata)

    @overload
    def __getitem__(self, idx: int) -> str:
        ...

    @overload
    def __getitem__(self, idx: list[int]) -> list[str]:
        ...

    def __getitem__(self, idx: int | list[int]) -> str | list[str]:
        # returns a genome name rather than actual data
        # this allows the collate_fn to get all data from .h5 disk file
        # referenced by self._data to read data once per batch, rather
        # than many times for each genome

        if isinstance(idx, int):
            idx = [idx]

        genomes = [self._genome_metadata.id2genome[index] for index in idx]
        if len(genomes) == 1:
            return genomes[0]
        return genomes

    def read_metadata(self, file: Path) -> GenomeMetadata:
        with file.open() as fp:
            genome2nptns = dict()
            genome2idx = dict()
            id2genome = dict()

            start = 0
            for id, line in enumerate(islice(fp, 1, None)):
                genome, _, _nptns = line.rstrip().split("\t")
                nptns = int(_nptns)
                end = start + nptns
                genome2idx[genome] = (start, end)
                genome2nptns[genome] = nptns
                id2genome[id] = genome
                start = end

        return GenomeMetadata(
            id2genome=id2genome,
            genome2nptns=genome2nptns,
            genome2idx=genome2idx,
        )

    def _convert_indices_to_slice(self, indices: list[tuple[int, int]]) -> slice:
        # sort by start first
        indices = sorted(indices, key=lambda x: x[0])
        # start of first genome
        start = indices[0][0]
        # end of last genome
        stop = indices[-1][1]
        return slice(start, stop)

    def collate_batch(self, batch: list[str]) -> torch.Tensor:
        """Receives as input from a DataLoader a list of the outputs from
        the `self.__getitem__` method, which returns genome names.
        This function converts all genome names in a batch to indices
        in the h5 data matrix to load in the part of the file
        that is needed.

        This is more efficient by reading the entire batch
        at once from disk, rather than allowing the dataloader to read
        each genome from disk.

        Then, the tensor is reshaped or padded as needed for a single contiguous
        3d tensor.

        Args:
            batch (list[str]): list of genome names

        Returns:
            torch.Tensor: 3d tensor [b, m, n]
                b: batch size
                m: max number of proteins
                n: protein embedding dimension
        """
        indices = list()
        genome_sizes = list()
        for genome in batch:
            indices.append(self._genome_metadata.genome2idx[genome])
            genome_sizes.append(self._genome_metadata.genome2nptns[genome])
        data_slice = self._convert_indices_to_slice(indices)
        X = torch.from_numpy(self._data[data_slice])

        if not all_equal(genome_sizes):
            # need to row-pad with 0s for a single contiguous 3d tensor
            tmp_X: list[torch.Tensor] = list()
            start = 0
            for size in genome_sizes:
                end = start + size
                tmp_X.append(X[start:end].clone())
                start = end
            X = pad_sequence(tmp_X, batch_first=True, padding_value=0.0)
        else:
            # no padding needed since all genomes are same size
            # just need to reshape the tensor
            n_genomes = len(batch)
            n_ptns = genome_sizes[0]
            feature_dim = X.size(-1)
            X = X.reshape(n_genomes, n_ptns, feature_dim)

        return X

    def convert_batch_idx_to_genome_ids(
        self, batch_idx: int, batch_size: int
    ) -> list[int]:
        start_genome_id = batch_idx * batch_size
        end_genome_id = (batch_idx + 1) * batch_size
        return list(range(start_genome_id, end_genome_id))

    def convert_batch_idx_to_genome_names(
        self, batch_idx: int, batch_size: int
    ) -> list[str]:
        genome_ids = self.convert_batch_idx_to_genome_ids(batch_idx, batch_size)
        return self[genome_ids]


class SimpleGenomeDataset(GenomeDataset):
    """Use when you can read the entire dataset into memory"""

    def __init__(self, data_file: Path, genome_metadata: Path) -> None:
        self._data: NDArray[float32] = tb.File(data_file, libver="latest").root.data[:]
        self._genome_metadata = self.read_metadata(genome_metadata)


@dataclass
class BatchInfo:
    n_batches: int
    batch_ids: torch.Tensor
    _remainder: int


def _batch_info(dataset_size: int, batch_size: int) -> BatchInfo:
    ##### floor div
    n_batches, _remainder = divmod(dataset_size, batch_size)
    batch_ids = torch.randperm(n_batches)
    return BatchInfo(n_batches, batch_ids, _remainder)


class WeakShuffleBatchSampler(Sampler):
    """Implements weak shuffling for datasets where individual data loading
    is inefficient, such as reading from h5 files on disk.

    Refer to: https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        n_batches: Optional[int] = None,
        batch_ids: Optional[torch.Tensor] = None,
    ) -> None:
        # TODO: need an alternate constructor for sampling from split batch indices
        # think easiest is to pass size of each split to first arg
        # then set batch_size to 1?
        # --> actually, need to accound for difference in position ie split: [10, 220, 2412]
        # get mapped to [0, 1, 2] -> suggests need a new init
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        if n_batches is None and batch_ids is None:
            # the last batch HAS to be done LAST
            # BUG: i think having self.n_batches have fewer is wrong and will break some things
            batch_info = _batch_info(self.dataset_size, self.batch_size)
            self.n_batches = batch_info.n_batches
            self.batch_ids = batch_info.batch_ids
            self._remainder = batch_info._remainder
        elif n_batches is not None and batch_ids is not None:
            self.n_batches = n_batches
            self.batch_ids = batch_ids
            self._set_remainder()
        else:
            raise ValueError(
                "Either both `n_batches` and `batch_ids` should be supplied or neither."
            )

    @property
    def num_batches(self) -> int:
        n_batches = self.n_batches
        if self._remainder > 0:
            n_batches += 1
        return n_batches

    def _set_remainder(self):
        n_full_batches, _r = divmod(self.dataset_size, self.batch_size)

        if _r > 0:
            max_batch_id = n_full_batches
            before_len = self.batch_ids.size()
            self.batch_ids = self.batch_ids[self.batch_ids != max_batch_id]
            after_len = self.batch_ids.size()

            if before_len == after_len:
                self._remainder = 0
            else:
                # no need to calc since just used as a signal
                self._remainder = 1
        else:
            self._remainder = 0

    def __len__(self) -> int:
        return self.batch_size

    def __iter__(self) -> Iterator[int]:
        for batch_id in self.batch_ids:
            start = int(batch_id * self.batch_size)
            end = int((batch_id + 1) * self.batch_size)
            yield from self._yield_data_idx(start, end)

        if self._remainder > 0:
            last_batch_id = self.dataset_size // self.batch_size
            start = last_batch_id * self.batch_size
            end = self.dataset_size
            yield from self._yield_data_idx(start, end)

    def _yield_data_idx(self, start: int, end: int) -> Iterator[int]:
        # Yield genome ids from batch ids
        indices = torch.arange(start, end)
        for idx in indices:
            yield int(idx)

    @classmethod
    def from_split(cls, dataset_size: int, split: torch.Tensor, batch_size: int):
        n_batches = int(split.size(0))
        return cls(
            dataset_size=dataset_size,
            batch_size=batch_size,
            n_batches=n_batches,
            batch_ids=split,
        )


### TODO: basic train / val splitter will work at the batch level?
# TODO: figure out if there are problems with this
# this could be good since data are prebatched by genome size, so this means
# that likely the most similar genomes are already grouped together,
# so this is sort of? like removing individual similarity groups
def _split_size(size: int, ratio: tuple[float, ...]) -> list[int]:
    sizes = [round(size * r) for r in ratio]
    diff = sum(sizes) - size
    if diff > 0:
        # rounded up -> take from back (ie test or val dataset)
        indices = cycle(reversed(range(len(sizes))))
        while sum(sizes) != size:
            idx = next(indices)
            sizes[idx] -= 1
    elif diff < 0:
        # rounded down -> add more to front (ie train dataset)
        indices = cycle(range(len(sizes)))
        while sum(sizes) != size:
            idx = next(indices)
            sizes[idx] += 1

    return sizes


class GenomeDataLoader:
    def __init__(
        self,
        dataset: GenomeDataset,
        batch_size: int,
        split_ratio: Optional[tuple[float, ...]] = None,
    ):
        self._dataset = dataset
        self.batch_size = batch_size
        self._batch_info = _batch_info(len(dataset), batch_size)

        # these will hold the batch level indices split into training splits
        self._splits: Optional[tuple[torch.Tensor, ...]] = None
        self._split_init(split_ratio)

    def _split_init(self, ratio: Optional[tuple[float, ...]]):
        if ratio is not None:
            if not math.isclose(sum(ratio), 1.0):
                raise ValueError(
                    f"Split ratio must add up to 1.0. Provided ratio ({ratio}) does not."
                )
            # ok so train / val datasets do not need to be contiguous,
            # but they do need to be sorted for optimal IO from disk

            if self._batch_info._remainder > 0:
                n_batches = self._batch_info.n_batches + 1
                batch_ids = torch.randperm(n_batches)
            else:
                n_batches = self._batch_info.n_batches
                batch_ids = self._batch_info.batch_ids
            split_sizes = _split_size(n_batches, ratio)
            _splits: tuple[torch.Tensor, ...] = batch_ids.split(split_sizes)
            # sort_idx = (s.sort()[1] for s in _splits)
            # self._splits = tuple(s[idx] for s, idx in zip(_splits, sort_idx))
            self._splits = _splits

    @staticmethod
    def _get_dataloader(
        dataset: GenomeDataset,
        batch_size: int,
        sampler: Optional[Sampler] = None,
        **kwargs,
    ) -> DataLoader:
        if sampler is None:
            sampler = WeakShuffleBatchSampler(len(dataset), batch_size)
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=dataset.collate_batch,
            sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
            **kwargs,
        )

    def get_dataloader(self, **kwargs) -> DataLoader:
        """Return a `DataLoader` specific to the on-disk `GenomeDataset` that will
        only read contiguously ordered batches.

        Returns:
            DataLoader: `torch.utils.data.DataLoader`
        """
        return self._get_dataloader(self._dataset, self.batch_size, **kwargs)

    def get_dataloaders(self, **kwargs) -> tuple[DataLoader, ...]:
        """Return as many `DataLoader`s specific to the on-disk `GenomeDataset` that will
        only read batches in order as requested by the `split_ratio` to __init__.

        Returns:
            tuple[DataLoader, ...]: `torch.utils.data.DataLoader`
        """
        if self._splits is None:
            raise ValueError(
                "Requires GenomeDataLoader to be initialized with a split ratio."
            )

        dataloaders = tuple(
            self._get_dataloader(
                dataset=self._dataset,
                batch_size=self.batch_size,
                sampler=WeakShuffleBatchSampler.from_split(
                    len(self._dataset), split, self.batch_size
                ),
                **kwargs,
            )
            for split in self._splits
        )
        return dataloaders


def ratio_split(
    data: GenomeDataset, ratio: tuple[float, ...] = (0.8, 0.2)
) -> tuple[list[int], ...]:
    """Simple ratio split that only selects contiguous pieces of data due to disk reading
    limitations.

    Args:
        data (GenomeDataset): dataset
        ratio (tuple[float, ...], optional): Ratio of split sizes. Defaults to (0.8, 0.2).

    Raises:
        ValueError: If ratio does not sum to 1.0

    Returns:
        tuple[torch.Tensor, ...]: Tensors with dataset indices
    """
    if not math.isclose(sum(ratio), 1.0):
        raise ValueError(
            f"Split ratio must add up to 1.0. Provided ratio ({ratio}) does not."
        )

    split_sizes = _split_size(len(data), ratio)
    split_sizes = random_sample(split_sizes, k=len(split_sizes))
    # take a random contiguous slice
    indices = torch.arange(len(data))
    splits: tuple[torch.Tensor, ...] = indices.split(split_sizes)
    return tuple(
        s.tolist() for s in sorted(splits, key=lambda x: int(x.size(0)), reverse=True)
    )


class GenomeSetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_file: Path,
        metadata_file: Path,
        batch_size: int = BATCH_SIZE,
        split_ratio: Optional[tuple[float, ...]] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self._data_file = data_file
        self._metadata_file = metadata_file
        self.batch_size = batch_size
        self._dataset = SimpleGenomeDataset(self._data_file, self._metadata_file)

        self.save_hyperparameters(ignore=["data_file", "metadata_file"])

    @property
    def feature_dimension(self) -> int:
        return self._dataset._data.shape[-1]

    def setup(self, stage: str):
        # each process gets this

        if stage == "fit":
            # TODO: add hyperparam to specific split size
            # TODO: probably want a better split than just random
            self._train_val_loader = GenomeDataLoader(
                self._dataset,
                self.batch_size,
                self.hparams["split_ratio"],
            )
            (
                self._train_loader,
                self._val_loader,
            ) = self._train_val_loader.get_dataloaders(
                num_workers=self.hparams["num_workers"],
                pin_memory=self.hparams["pin_memory"],
            )

        if stage == "test":
            self.test_dataset = self._dataset

        if stage == "predict":
            self.predict_dataset = self._dataset

    def _dataloader(self, dataset: GenomeDataset, **kwargs) -> DataLoader:
        dataloader = GenomeDataLoader(dataset, self.batch_size).get_dataloader(
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    # TODO: these don't want the weak shuffling
    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(self.predict_dataset, shuffle=False)

import math
import warnings
from operator import add
from typing import Callable, Iterable, Iterator, Literal, Optional, Sequence, cast

from torch import Generator, default_generator, randperm
from torch.utils.data import Subset
from torch.utils.data import random_split as scaffold_random_split

from pst.data.dataset import GenomeDataset, SubsetGenomeDataset
from pst.typing import _T


# Taken from python 3.5 docs
def _accumulate(iterable: Iterable[_T], fn: Callable[[_T, _T], _T] = add) -> Iterator[_T]:
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def genome_random_split(
    dataset: GenomeDataset,
    lengths: Sequence[int | float],
    generator: Optional[Generator] = default_generator,
) -> list[SubsetGenomeDataset]:
    # this is the same implementation as pytorch's random_split except using the
    # GenomeDataset.num_genomes for dataset size
    dataset_len = dataset.num_genomes
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(dataset_len * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = dataset_len - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != dataset_len:  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    splits = [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]

    return cast(list[SubsetGenomeDataset], splits)


def random_split(
    dataset: GenomeDataset,
    lengths: Sequence[int | float],
    split_level: Literal["scaffold", "genome"] = "scaffold",
    generator: Optional[Generator] = default_generator,
) -> list[SubsetGenomeDataset]:
    if split_level == "scaffold":
        fn = scaffold_random_split
    else:
        fn = genome_random_split

    splits = fn(dataset, lengths, generator)

    return cast(list[SubsetGenomeDataset], splits)

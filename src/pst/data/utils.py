from collections.abc import Iterable
from typing import NamedTuple, TypedDict

import tables as tb
import torch
from torch_geometric.utils import scatter

from pst.data._types import FeatureLevel

H5_FILE_COMPR_FILTERS = tb.Filters(complevel=4, complib="blosc:lz4")  # type: ignore


def graph_sizes_to_index_pointer(sizes: torch.Tensor) -> torch.Tensor:
    """For multiple graphs who node embeddings are stacked into a batch, this will convert a
    tensor of graph sizes to an index pointer tensor that can be used to index into the batch
    tensor to get the all node embeddings for each graph.

    This is the CSR format of a graph used by PyTorch Geometric.

    Args:
        sizes (torch.Tensor): A 1D tensor of graph sizes, shape: [N graphs]

    Returns:
        torch.Tensor: A 1D tensor of index pointers, shape: [N graphs + 1]. To get the start
            and stop indices for the i-th graph, use index_pointer[i] and index_pointer[i + 1].
    """
    ptr = torch.zeros(sizes.size(0) + 1, dtype=torch.long)
    ptr[1:] = sizes.cumsum(0)

    return ptr


def graph_batch_label_to_index_pointer(batch: torch.Tensor) -> torch.Tensor:
    """For multiple graphs who node embeddings are stacked into a batch, this will convert a
    tensor of graph labels to an index pointer tensor that can be used to index into the batch
    tensor to get the all node embeddings for each graph.

    Args:
        batch (torch.Tensor): A 1D tensor of graph labels, shape: [N nodes], for batched
            graphs that specifies which graph each node belongs to.

    Returns:
        torch.Tensor: A 1D tensor of index pointers, shape: [N graphs + 1]. To get the start
            and stop indices for the i-th graph, use index_pointer[i] and index_pointer[i + 1].
    """
    sizes = torch.bincount(batch)
    return graph_sizes_to_index_pointer(sizes)


def convert_to_scaffold_level_genome_label(
    genome_label: torch.Tensor, scaffold_label: torch.Tensor
) -> torch.Tensor:
    """For genome and scaffold labelings of sub-scaffold genomic chunks, this function will
    convert the genome labeling to a scaffold-level labeling that can be used for a 2-step
    embedding reduction of chunk-level embeddings to scaffold-level to genome-level.

    More specifically, for a `GenomeDataset` object, the inputs to this function should be of
    shape [N 'scaffolds'], since what is considered a scaffold by the dataset may actually be
    an artificially chunked scaffold created from the `GenomeDataset.fragment` method.
    These tensors should map subchunks to genomes or scaffolds but also accounts
    for naturally fragmented genomes or metagenomes.

    However, the forward pass of a `ProteinSetTransformer` will produce chunk-level genomic
    embeddings. To reduce these embeddings to scaffold-level embeddings, we can just do this:

    ```python
    from torch_geometric.utils import scatter

    # chunk_embeddings -> shape: [N chunks, D]
    # scaffold_label -> shape: [N chunks]

    # output shape: [N scaffolds, D]
    scaffold_embeddings = scatter(chunk_embeddings, scaffold_label, reduce="mean")
    ```

    This introduces a problem since the original genome_label tensor is the same shape as the
    the number of chunks, but the scaffold_embeddings tensor can be smaller. Thus, we need to
    reduce the genome_label tensor to refer to scaffolds instead of sub-scaffold chunks.

    For example, suppose we have 3 genomes composed of 2 scaffolds each that were artificially
    chunked into 2 sub-scaffolds each. We have 12 total genomic chunks. Then
    scaffold_label=[0 0 1 1 2 2 3 3 4 4 5 5] shape: [12]
    genome_label = [0 0 0 0 1 1 1 1 2 2 2 2] shape: [12]
    Reducing the chunk-level embeddings to the scaffold level will change the shape of the
    embedding tensor to [6, D] since there were originally 6 real scaffolds. To further
    reduce the scaffold-level embeddings to genome-level, we need to change the genome_label
    tensor to look like this: [0 0 1 1 2 2] shape: [6] using this function.

    Examples:
    ```python
    import torch

    >>> genome_label = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    >>> scaffold_label = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    >>> convert_to_scaffold_level_genome_label(genome_label, scaffold_label)
    tensor([0, 0, 1, 1, 2, 2])
    ```

    Args:
        genome_label (torch.Tensor): 1d tensor of genome labels, shape: [N chunks]
            labels each chunk with a genome ID
        scaffold_label (torch.Tensor): 1d tensor of scaffold labels, shape: [N chunks]
            labels each chunk with a scaffold ID

    Raises:
        ValueError: if the number of elements in genome_label and scaffold_label are not equal

    Returns:
        torch.Tensor: scaffold-level genome label tensor, shape: [N scaffolds]
    """
    if genome_label.numel() != scaffold_label.numel():
        raise ValueError("Genome label and scaffold label must have the same number of elements")

    return scatter(genome_label, scaffold_label, reduce="any")


def compute_group_frequency_weights(
    groups: torch.Tensor, log_inverse: bool = False
) -> torch.Tensor:
    """Compute frequency-based weights for each group in a tensor of group labels. The weights are based on inverse frequency.

    Args:
        groups (torch.Tensor): 1D integer tensor of group labels
        log_inverse (bool, optional): Whether to take the log of the inverse frequency. Defaults to False.

    Returns:
        torch.Tensor: 1D tensor of weights for each group
    """
    group_counts: torch.Tensor
    _, inverse_index, group_counts = torch.unique(groups, return_inverse=True, return_counts=True)

    freq = group_counts / group_counts.sum()
    inv_freq = 1 / freq
    if log_inverse:
        # major class imbalance will lead to extremely rare groups dominate reweighting
        inv_freq = torch.log(inv_freq)

    # not really sure if needed since relative contribution to loss should be the same
    inv_freq /= torch.amin(inv_freq)

    weights = inv_freq[inverse_index]
    return weights


def _merge_indices(indices: Iterable[int]) -> list[slice]:
    """Merge a list of indices into a list of slices. This converts individual indices into
    slices when they are contiguous. For example, the indices [0, 1, 2, 4, 5, 6] will be
    converted to [slice(0, 3), slice(4, 7)], which can be used for list/array slicing.

    Args:
        indices (Iterable[int]): A list of indices to merge.

    Returns:
        list[slice]: A list of slices representing the merged indices.

    Example:
        >>> _merge_indices([0, 1, 2, 4, 5, 6])
        [slice(0, 3), slice(4, 7)]
        >>> _merge_indices([0, 2, 3, 4, 5])
        [slice(0, 1), slice(2, 6)]
    """
    indices = sorted(set(indices))
    if not indices:
        return []

    if len(indices) == 1:
        return [slice(indices[0], indices[0] + 1)]

    slices: list[slice] = []
    start = indices[0]
    stop = start + 1

    prev = start
    for curr in indices[1:]:
        diff = curr - prev
        prev = curr
        if diff == 1:
            stop += 1
        else:
            slices.append(slice(start, stop))
            start = curr
            stop = start + 1

    # get the last slice
    slices.append(slice(start, stop))
    return slices


class RegisteredFeature(NamedTuple):
    name: str
    data: torch.Tensor
    feature_level: FeatureLevel


class _ScaffoldFeatureFragmentedData(TypedDict):
    scaffold_part_of_multiscaffold: torch.Tensor
    scaffold_registry: list[RegisteredFeature]


class _FragmentedData(_ScaffoldFeatureFragmentedData):
    scaffold_sizes: torch.Tensor
    scaffold_ptr: torch.Tensor
    scaffold_genome_label: torch.Tensor
    scaffold_label: torch.Tensor

from copy import deepcopy
from typing import cast

import torch

from pst.data.utils import graph_sizes_to_index_pointer
from pst.typing import GenomeGraphBatch, MaskedGenomeGraphBatch, OptTensor


def create_node_mask(
    masking_rate: float, sizes: torch.Tensor, ptr: OptTensor = None
) -> torch.Tensor:
    # sizes shape: [num scaffolds]
    # ptr shape:   [num scaffolds + 1]
    # mask shape:  [num proteins]

    if masking_rate < 0 or masking_rate > 0.5:
        raise ValueError("masking_rate should be in [0.0, 0.5]")

    if ptr is None:
        ptr = graph_sizes_to_index_pointer(sizes)

    mask = torch.zeros(int(sizes.sum()), device=sizes.device, dtype=torch.bool)
    for idx, size in enumerate(sizes):
        view = mask[ptr[idx] : ptr[idx + 1]]
        # guarantee that at least one element is masked
        while True:
            scaffold_mask: torch.Tensor = torch.rand(size, device=sizes.device) < masking_rate  # type: ignore
            if scaffold_mask.any():
                break

        view[:] = scaffold_mask

    return mask


def mask_batch(
    batch: GenomeGraphBatch, masking_rate: float, copy: bool = False
) -> MaskedGenomeGraphBatch:
    """Mask protein nodes in a batch of genome graphs.
    This is a simpler wrapper that can easily be opted in without a custom DataLoader.

    Args:
        batch (GenomeGraphBatch): batch of genome graphs generated from a `GenomeDataset`
        masking_rate (float): masking rate for the protein nodes. Must be in [0.0, 0.5]
        copy (bool, optional): whether to copy the batch before masking. Defaults to False, meaning the batch is modified in place.

    Returns:
        MaskedGenomeGraphBatch: the same as a GenomeGraphBatch but guaranteed to have `masked_embeddings`
            field for the masked protein embeddings without position or strand information and
            a `node_mask` field for the boolean mask of the masked nodes.
    """
    node_mask = create_node_mask(masking_rate, batch.num_proteins, batch.ptr)
    if copy:
        batch = deepcopy(batch)
    masked_embeddings_wo_pos_or_strand = batch.x[node_mask]
    batch.x[node_mask] = 0.0

    setattr(batch, "masked_embeddings", masked_embeddings_wo_pos_or_strand)
    setattr(batch, "node_mask", node_mask)

    return cast(MaskedGenomeGraphBatch, batch)

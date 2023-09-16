import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import segment


def standardize(
    x: Tensor,
    ptr: OptTensor = None,
    batch: OptTensor = None,
    size: OptTensor = None,
    strict: bool = True,
) -> Tensor:
    """
    Standardize a stacked batch tensor per graph/set.

    Args:
        x (torch.Tensor): [N, D] Input stacked node/item features.
        ptr (torch.Tensor, optional): Points to the start/stop of each graph/set
            in the input `:param:x` stacked features. `ptr[i]` points to
            the starting index of graph/set i, and `ptr[i+1]` points to
            the end. If not provided, it is calculated from the batch tensor.
        batch (torch.Tensor, optional): [N] LongTensor encoding the graph/set
            each node/item belongs to. Only required if standardizing inputs.
        size (torch.Tensor, optional): [N] LongTensor encoding the number of
            nodes/items per graph/set. If not provided, it is calculated from
            the batch tensor.

    Returns:
        Tensor: [N, D] Standardized node/item features.
    """
    if batch is None:
        if strict:
            raise RuntimeError(
                (
                    "Must pass batch tensor when standardizing in strict mode. "
                    "If all nodes belong to the same graph, use strict=False."
                )
            )
        # assume all nodes part of same graph
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if size is None:
        size = torch.bincount(batch)

    size = size.unsqueeze(-1)

    if ptr is None:
        ptr = torch.cat([size.new_zeros(1), size.cumsum(dim=0)])

    # shape: [batch_size, D]
    per_graph_mean = segment(x, ptr, reduce="mean")
    # expand to match x shape: [N, D]
    per_graph_mean = per_graph_mean[batch]

    # manually calc std
    summand = segment(torch.square(x - per_graph_mean), ptr=ptr, reduce="sum")
    per_graph_std = torch.sqrt(summand / (size - 1))
    zero_tensor = per_graph_std.new_zeros(1)

    # avoid division by zero when std is 0
    per_graph_std = torch.where(per_graph_std.isclose(zero_tensor), 1.0, per_graph_std)
    # expand to match x shape: [N, D]
    per_graph_std = per_graph_std[batch]

    # standardize
    x = (x - per_graph_mean) / per_graph_std
    return x

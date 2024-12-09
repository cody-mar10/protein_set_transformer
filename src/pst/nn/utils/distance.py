import torch
from einops import reduce
from torch_scatter import segment_mean_csr, segment_min_csr

from pst.typing import OptTensor, PairTensor


def pairwise_euclidean_distance(x: torch.Tensor, y: OptTensor = None) -> torch.Tensor:
    """Returns squared Euclidean distance between two sets of points.

    Args:
        x (torch.Tensor): point tensor of shape [N, d]
        y (OptTensor, optional): point tensor of shape [M, d]. Defaults to None. If None,
            y = x, and the diagonal (self-comparisons) will be set to 0.0.

    Returns:
        torch.Tensor: squared Euclidean distance tensor of shape [N, M]
    """
    if y is None:
        y = x
        dist = torch.cdist(x, y, p=2.0).square().fill_diagonal_(0.0)
    else:
        dist = torch.cdist(x, y, p=2.0).square()

    return dist


def _stacked_batch_chamfer_distance(
    all_pairwise_dist: torch.Tensor, ptr: torch.Tensor
) -> PairTensor:
    min_dists, flow_idx = segment_min_csr(all_pairwise_dist, ptr)
    mean_dist = segment_mean_csr(min_dists.t(), ptr)

    chamfer_dist = mean_dist + mean_dist.t()
    return chamfer_dist, flow_idx.t()


# TODO: allow returning the min_distances instead of the chamfer distance
# will be useful for speeding up augmented negative sampling
def stacked_batch_chamfer_distance(
    batch: torch.Tensor, ptr: torch.Tensor, *, other: OptTensor = None
) -> PairTensor:
    """Compute Chamfer distance for all point sets stacked into a 2D batch
    without padding.

    Args:
        batch (torch.Tensor): tensor of all item features concatenated into
            a 2D tensor of shape [N, d], where N is the total number of
            items in all point sets and d is the item embedding dimension.
        ptr (torch.Tensor): contains pointers to the start of each point
            set in `batch`. ptr[i] is the start of point set i, and ptr[i+1]
            is the end of point set i.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:

            - chamfer distance tensor, shape [batch_size, batch_size]
            - item flow indices, F, shape [N, batch_size]
              - Fij points to the item index in `batch` that is closest to
              the ith them in point set j.
    """

    # if other is None: chamfer distance for all graphs/sets in the batch
    # else: chamfer distance between real and augmented point sets
    all_pairwise_dist = pairwise_euclidean_distance(batch, other)
    return _stacked_batch_chamfer_distance(all_pairwise_dist, ptr)


def pairwise_chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    dist = torch.cdist(x, y, p=2.0).square()
    x_min = reduce(dist, "x y -> x", "min")
    y_min = reduce(dist, "x y -> y", "min")

    chamfer_dist = float(x_min.mean() + y_min.mean())
    return chamfer_dist

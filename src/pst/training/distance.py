from __future__ import annotations

import torch


# this will only get compiled when a CUDA GPU is available
@torch.compile(mode="max-autotune", disable=not torch.cuda.is_available())
def stacked_batch_chamfer_distance(
    batch: torch.Tensor, ptr: torch.Tensor
) -> tuple[torch.Tensor, dict[tuple[int, int], torch.Tensor]]:
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
        tuple[torch.Tensor, dict[tuple[int, int], torch.Tensor]]:
            chamfer distance tensor and the item flow indices for each
            combination of point sets
    """
    all_pairwise_dist = torch.cdist(batch, batch, p=2.0).square().fill_diagonal_(0.0)
    batch_size = int(ptr.numel() - 1)
    chamfer_dist = torch.zeros(batch_size, batch_size, device=batch.device)

    # TODO: this could probably just be a single tensor, and then you compute the offsets as necessary
    flow_idx: dict[tuple[int, int], torch.Tensor] = dict()
    for i in range(batch_size):
        start_x = ptr[i]
        end_x = ptr[i + 1]

        for j in range(i + 1, batch_size):
            start_y = ptr[j]
            end_y = ptr[j + 1]

            block = all_pairwise_dist[start_x:end_x, start_y:end_y]

            x_min: torch.Tensor
            x_flow: torch.Tensor
            y_min: torch.Tensor
            y_flow: torch.Tensor
            x_min, x_flow = block.min(dim=1)
            y_min, y_flow = block.min(dim=0)

            x_dist = x_min.mean()
            y_dist = y_min.mean()
            dist = x_dist + y_dist
            chamfer_dist[i, j] = dist
            chamfer_dist[j, i] = dist

            flow_idx[i, j] = x_flow
            flow_idx[j, i] = y_flow

    return chamfer_dist, flow_idx

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load
from torch_geometric.utils import segment

stacked_batch_chamfer_distance_cpu = load(
    "stacked_batch_chamfer_distance_cpu",
    sources=["stacked_batch_chamfer_distance_cpu.cpp"],
    extra_cflags=["-O2"],
    verbose=True,
    build_directory=".",
)

if torch.cuda.is_available():
    stacked_batch_chamfer_distance_cuda = load(
        "stacked_batch_chamfer_distance_cuda",
        sources=[
            "stacked_batch_chamfer_distance_cuda.cpp",
            "stacked_batch_chamfer_distance_cuda_kernel.cu",
        ],
        extra_cflags=["-O2"],
        verbose=True,
        build_directory=".",
    )


def stacked_batch_chamfer_distance(
    x: torch.Tensor, ptr: torch.Tensor, sizes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Computes the Chamfer distance between a batch of point sets.

    Args:
        x (Tensor): The input point sets. Its shape should be
            :obj:`[sum(sizes), num_dims]`.
        ptr (LongTensor): The index pointer for each point set in :obj:`x`.
            Its shape should be :obj:`[batch_size + 1]`.
        sizes (LongTensor): The size of each point set in :obj:`x`. Its shape
            should be :obj:`[batch_size]`.

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """

    if x.is_cuda:
        func = stacked_batch_chamfer_distance_cuda.stacked_batch_chamfer_distance_cuda  # type: ignore # noqa
    else:
        func = stacked_batch_chamfer_distance_cpu.stacked_batch_chamfer_distance_cpu  # type: ignore # noqa

    min_distances, flow = func(x, ptr, sizes)

    chamfer_distance = segment(min_distances, ptr=ptr, reduce="mean")
    chamfer_distance = chamfer_distance + chamfer_distance.transpose(-2, -1)
    return chamfer_distance, flow

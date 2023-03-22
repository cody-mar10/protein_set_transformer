from __future__ import annotations

from itertools import combinations
from typing import Literal, Optional

import torch
from pst.utils import DistFuncSignature, FlowDict

DISTANCE_TYPES = Literal["cosine", "angular", "euclidean"]


def _cosine_similarity(
    X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    # compute norm along rows ie per protein
    # add small constant to prevent nans
    norm_X: torch.Tensor = torch.linalg.norm(X, dim=1) + eps
    norm_Y: torch.Tensor = torch.linalg.norm(Y, dim=1) + eps
    X = X / norm_X.unsqueeze(1)
    Y = Y / norm_Y.unsqueeze(1)
    return torch.matmul(X, Y.t())


def cosine_distance(
    X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    return 1 - _cosine_similarity(X, Y, eps)


def angular_distance(
    X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    similarity = _cosine_similarity(X, Y, eps)
    distance = torch.arccos(similarity) / torch.pi
    return distance


def euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # X and Y are both 2d matrices
    # this will calculate the distance between each vector (row) from X and Y
    distance = torch.cdist(X, Y, p=2)
    return distance


class SetDistance:
    def __init__(self, distfunc: DistFuncSignature = euclidean_distance) -> None:
        self.distfunc = distfunc

    def clear(self):
        # map (Xi, Xj) to flow from Xim -> Xjn
        # for all pairs of X in minibatch
        self._flow: FlowDict = dict()
        # pairwise relaxed word movers distance between all sets
        # in minibatch
        self._emd = torch.tensor(0.0)

    @property
    def flow(self) -> FlowDict:
        return self._flow

    @property
    def emd(self) -> torch.Tensor:
        return self._emd

    def fit_transform(
        self, X: torch.Tensor, Y: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, FlowDict]:
        self.fit(X, Y)
        return self._emd, self._flow

    def fit(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        # initialize self._flow and self._emd
        self.clear()
        if Y is not None:
            dist, flow_ij, flow_ji = self._fit_return(X, Y)
            self._emd = dist
            self._flow[0, 1] = flow_ij
            self._flow[1, 0] = flow_ji
        elif X.dim() > 2:
            # calc distance between all datasets in a batch
            batch_size = X.size(0)
            self._emd = (
                torch.empty((batch_size, batch_size), device=X.device)
                # prevent self-comparison from being chosen for positive sampling
                .fill_diagonal_(float("inf"))
            )
            indices = combinations(range(batch_size), r=2)

            for i, j in indices:
                dist, flow_ij, flow_ji = self._fit_return(X[i], X[j])
                self._flow[i, j] = flow_ij
                self._flow[j, i] = flow_ji
                self._emd[i, j] = dist
                self._emd[j, i] = dist
        else:
            # only passed X and X is just a 1D or 2D tensor
            dist, flow_ij, _ = self._fit_return(X, X)
            self._emd = dist
            self._flow[0, 0] = flow_ij

    def _fit_return(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pairwise cost for all elements in Xi and Xj
        cost = self._cost_matrix(X, Y)

        # flow is not symmetric
        flow_ij = self._relaxed_flow(cost)
        flow_ji = self._relaxed_flow(cost.t())

        # dist between sets Xi and Xj
        # flow_ij.sum() == X.size(-2) ie n_elements
        dist = torch.sum(cost * flow_ij / flow_ij.sum())
        return dist, flow_ij.nonzero(), flow_ji.nonzero()

    def _cost_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.distfunc(X, Y).to(device=X.device)

    def _relaxed_flow(self, cost: torch.Tensor) -> torch.Tensor:
        # put all flow on most similar vector across all pairs of
        # vectors in (Xi, Xj)
        # flow from Xim -> Xjn
        # to get Xjn -> Xim just transpose cost
        flow = torch.zeros_like(cost, device=cost.device)
        x_idx = torch.arange(cost.size(0), device=cost.device)
        # min cost
        # ex: X = [[0, 1, 2], [3, 4, 5]]
        # X.min(0)[0] == [0, 1, 2]
        # X.min(1)[0] == [0, 3]
        y_idx = cost.min(dim=1)[1]
        flow[x_idx, y_idx] = 1.0
        return flow

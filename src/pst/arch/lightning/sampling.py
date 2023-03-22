from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from pst.arch.lightning.distance import (
    SetDistance,
    euclidean_distance,
    cosine_distance,
    angular_distance,
    DISTANCE_TYPES,
)
from pst.utils import DistFuncSignature, FlowDict


@dataclass
class TripletSample:
    idx: torch.Tensor
    weights: torch.Tensor


@dataclass
class AugmentedSample:
    data: torch.Tensor
    weights: torch.Tensor
    negative_idx: torch.Tensor


class TripletSampler:
    _SMALL_RANGE_DISTANCES = {"cosine", "angular"}

    def __init__(
        self,
        emd: torch.Tensor,
        disttype: DISTANCE_TYPES = "euclidean",
        scale: float = 7.0,
    ):
        self._emd = emd
        # rowwise min, ie most similar set per row
        self._min: tuple[torch.Tensor, torch.Tensor] = self._emd.min(dim=0)
        self._device = self._emd.device
        self._scale = scale
        self._disttype = disttype

    def _positive_sampling(self) -> torch.Tensor:
        # return indices of most similar set per row
        return self._min[1]

    def _negative_sampling(self, emd: Optional[torch.Tensor] = None) -> torch.Tensor:
        emd = emd if emd is not None else self._emd
        diag_mask = torch.eye(
            emd.size(0), dtype=torch.bool, device=self._device
        ).logical_not()

        # can't choose min and can't choose self
        negative_sample_space = torch.where(
            (emd > self._min[0].unsqueeze(1)) & (diag_mask), 1, 0
        )
        return torch.multinomial(negative_sample_space.float(), 1).squeeze()

    def _negative_weights(
        self,
        anchor_idx: torch.Tensor,
        negative_sample_idx: torch.Tensor,
        emd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        indices = (anchor_idx, negative_sample_idx)
        emd = emd if emd is not None else self._emd
        # diagonals are infinity, so need to ignore

        if self._disttype in self._SMALL_RANGE_DISTANCES:
            # changed to mean from std, since cosine dist
            # has a small range unlike euclidean distance
            scaled_emd = emd[torch.where(emd != float("inf"))].mean()
        else:
            scaled_emd = emd[torch.where(emd != float("inf"))].std()
        denom = 2 * (self._scale * scaled_emd) ** 2
        weights = torch.exp(-emd[indices] / denom)
        return weights

    def triplet_sampling(self) -> TripletSample:
        positive_sample_idx = self._positive_sampling()
        negative_sample_idx = self._negative_sampling()
        anchor_idx = torch.arange(self._emd.size(0), device=self._device)
        # [3, b] -> (anchor, positive sample, negative sample)
        samples = torch.vstack(
            (anchor_idx, positive_sample_idx, negative_sample_idx)
        ).long()
        weights = self._negative_weights(anchor_idx, negative_sample_idx)
        return TripletSample(samples, weights)


class PointSwapSampler(TripletSampler):
    _SENTINEL = -1
    _DISTFUNC2TYPE: dict[DistFuncSignature, DISTANCE_TYPES] = {
        cosine_distance: "cosine",
        angular_distance: "angular",
        euclidean_distance: "euclidean",
    }

    def __init__(
        self,
        emd: torch.Tensor,
        batch: torch.Tensor,
        flow: FlowDict,
        row_mask: torch.Tensor,
        sample_rate: float,
        scale: float = 7.0,
        distfunc: DistFuncSignature = euclidean_distance,
    ):
        super().__init__(emd=emd, disttype=self._DISTFUNC2TYPE[distfunc], scale=scale)
        self._batch = batch
        self._flow = flow
        self._sample_rate = sample_rate
        # indicates which rows are real ptns since batch is row-padded
        self._row_mask = row_mask
        self._anchor_idx = set(range(self._row_mask.size(-1)))
        self._rwmdistance = SetDistance(distfunc)
        self._triplet_sample = self.triplet_sampling()
        self._positive_sample = self._triplet_sample.idx[0:2]
        # used for indexing flow dict
        self._positive_idx = [(int(i), int(j)) for i, j in self._positive_sample.t()]
        self._point_swap_idx = self._build_point_swap_idx()

    def _build_point_swap_idx(self) -> torch.Tensor:
        max_samples = torch.ceil(self._row_mask[-1].sum() * self._sample_rate)
        samples: list[torch.Tensor] = list()
        for row in self._row_mask:
            n_samples = torch.ceil(row.sum() * self._sample_rate)
            padsize = int(max_samples - n_samples)
            sample = torch.multinomial(row.float(), int(n_samples))
            # TODO: may be able to replace with pad_sequence
            samples.append(F.pad(sample, (0, padsize), value=self._SENTINEL))
        return torch.stack(samples)

    def _point_swap(self, i: int, j: int) -> torch.Tensor:
        mask = self._point_swap_idx[i] != self._SENTINEL
        point_swap_idx = self._point_swap_idx[i, mask]

        # 0 col is for Xi, 1 col is for Xj
        # select rows corresponding to ptns to be swapped
        chosen_ptns = self._flow[i, j][point_swap_idx]
        chosen_self = set(map(int, chosen_ptns[:, 0]))
        not_chosen_self_set = self._anchor_idx - chosen_self
        swapped = self._batch[j][chosen_ptns[:, 1]]

        if not_chosen_self_set:
            # this is strictly for viruses with 1 gene
            not_chosen_self = torch.tensor(list(not_chosen_self_set))
            not_swapped = self._batch[i][not_chosen_self]

            # TODO: this will be ok with set permutation invariance
            # but if wanting to encode position, need to be careful
            # FIX: make this and then reorder it based on a particular idx
            # positional encodings get added during forward method of model
            return torch.vstack((not_swapped, swapped))
        return swapped

    def point_swap_sampling(self) -> AugmentedSample:
        augmented_samples = [self._point_swap(i, j) for i, j in self._positive_idx]
        X_aug = torch.stack(augmented_samples)
        aug_emd, _ = self._rwmdistance.fit_transform(X_aug)
        aug_negative_idx = self._negative_sampling(aug_emd)
        aug_weights = self._negative_weights(
            self._triplet_sample.idx[0], aug_negative_idx, aug_emd
        )
        return AugmentedSample(
            data=X_aug,
            weights=aug_weights,
            negative_idx=aug_negative_idx,
        )

    def sample(self) -> tuple[TripletSample, AugmentedSample]:
        return self._triplet_sample, self.point_swap_sampling()

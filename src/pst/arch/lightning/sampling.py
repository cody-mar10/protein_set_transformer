from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pst.arch.lightning.data import GenomeSetDataModule
from pst.arch.lightning.distance import (
    SetDistance,
    euclidean_distance,
    cosine_distance,
    angular_distance,
    DISTANCE_TYPES,
)
from pst.utils import DistFuncSignature, FlowDict
from pst.utils.mask import compute_row_mask


@dataclass
class TripletSample:
    idx: torch.Tensor
    weights: torch.Tensor

    def to(self, device: torch.device) -> TripletSample:
        self.idx = self.idx.to(device=device)
        self.weights = self.weights.to(device=device)
        return self


@dataclass
class AugmentedSample:
    data: torch.Tensor
    weights: torch.Tensor
    negative_idx: torch.Tensor

    def to(self, device: torch.device) -> AugmentedSample:
        self.data = self.data.to(device=device)
        self.weights = self.weights.to(device=device)
        self.negative_idx = self.negative_idx.to(device=device)
        return self


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

    def _negative_sampling(self) -> torch.Tensor:
        diag_mask = torch.eye(
            self._emd.size(0), dtype=torch.bool, device=self._device
        ).logical_not()

        # can't choose min and can't choose self
        negative_sample_space = torch.where(
            (self._emd > self._min[0].unsqueeze(1)) & (diag_mask), 1, 0
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


class PointSwapSampler:
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
        self._batch = batch
        self._flow = flow
        self._sample_rate = sample_rate
        self._scale = scale
        self._distfunc = distfunc
        # indicates which rows are real ptns since batch is row-padded
        self._row_mask = row_mask
        self._anchor_idx = set(range(self._row_mask.size(-1)))
        self._rwmdistance = SetDistance(distfunc)
        self._triplet_sampler = TripletSampler(
            emd=emd,
            disttype=self._DISTFUNC2TYPE[distfunc],
            scale=scale,
        )
        self._triplet_sample = self._triplet_sampler.triplet_sampling()
        self._positive_sample = self._triplet_sample.idx[0:2]
        # used for indexing flow dict
        self._positive_idx = [(int(i), int(j)) for i, j in self._positive_sample.t()]
        self._point_swap_idx = self._build_point_swap_idx()

    def _build_point_swap_idx(self) -> torch.Tensor:
        max_samples = torch.ceil(self._row_mask[-1].sum() * self._sample_rate)
        samples: list[torch.Tensor] = list()
        for row in self._row_mask:
            # row = row.squeeze(dim=0)
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
        aug_triplet_sampler = TripletSampler(
            emd=aug_emd,
            disttype=self._DISTFUNC2TYPE[self._distfunc],
            scale=self._scale,
        )
        aug_negative_idx = aug_triplet_sampler._negative_sampling()
        aug_weights = aug_triplet_sampler._negative_weights(
            self._triplet_sample.idx[0], aug_negative_idx, aug_emd
        )
        return AugmentedSample(
            data=X_aug,
            weights=aug_weights,
            negative_idx=aug_negative_idx,
        )

    def sample(self) -> tuple[TripletSample, AugmentedSample]:
        return self._triplet_sample, self.point_swap_sampling()


def precompute_point_swap_sampling(
    dataloader: Optional[DataLoader] = None,
    datamodule: Optional[GenomeSetDataModule] = None,
    sample_rate: float = 0.5,
    scale: float = 7.0,
    distfunc: DistFuncSignature = euclidean_distance,
):
    def _precompute(dataloader: DataLoader) -> dict[str, torch.Tensor]:
        RWMDDistance = SetDistance(distfunc=distfunc)
        triplet_sample_indices: list[torch.Tensor] = list()
        triplet_sample_weights: list[torch.Tensor] = list()
        aug_sample_data: dict[int, torch.Tensor] = dict()
        aug_sample_neg_indices: list[torch.Tensor] = list()
        aug_sample_weights: list[torch.Tensor] = list()
        for batch_idx, batch in enumerate(dataloader):
            row_mask = compute_row_mask(batch)
            rwmd, flow = RWMDDistance.fit_transform(batch)
            sampler = PointSwapSampler(
                emd=rwmd,
                batch=batch,
                flow=flow,
                row_mask=row_mask,
                sample_rate=sample_rate,
                scale=scale,
                distfunc=distfunc,
            )
            triplet_sample, aug_sample = sampler.sample()
            # TODO: move to cpu and then to device?
            triplet_sample = triplet_sample.to(torch.device("cpu"))
            aug_sample = aug_sample.to(torch.device("cpu"))

            triplet_sample_indices.append(triplet_sample.idx)
            triplet_sample_weights.append(triplet_sample.weights)
            aug_sample_data[batch_idx] = aug_sample.data
            aug_sample_neg_indices.append(aug_sample.negative_idx)
            aug_sample_weights.append(aug_sample.weights)

        samples = {
            "triplet": {
                "indices": torch.stack(triplet_sample_indices),
                "weights": torch.stack(triplet_sample_weights),
            },
            "aug": {
                "data": aug_sample_data,
                "weights": torch.stack(aug_sample_weights),
                "negative_indices": torch.stack(aug_sample_neg_indices),
            },
        }
        return samples

    if datamodule is not None:
        datamodule.setup("predict")
        dataloader = datamodule.predict_dataloader()
        precomputed_results = _precompute(dataloader)
    elif dataloader is not None:
        precomputed_results = _precompute(dataloader)
    else:
        if dataloader is None and datamodule is None:
            raise ValueError(
                "Either a dataloader or a lightning data module is required"
            )
        else:
            raise ValueError(
                "Dataloader and datamodule are mutually exclusive. Only provide one."
            )

    return precomputed_results

from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pst.arch.lightning.distance import (
    SetDistance,
    euclidean_distance,
    cosine_distance,
    angular_distance,
    DISTANCE_TYPES,
)
from pst.utils import DistFuncSignature, FlowDict
from pst.utils.cli import Args
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


class PositiveSampler:
    def __init__(
        self,
        emd: torch.Tensor,
    ):
        self._emd = emd
        # rowwise min, ie most similar set per row
        self._min: tuple[torch.Tensor, torch.Tensor] = self._emd.min(dim=0)
        self._device = self._emd.device

    def _positive_sampling(self) -> torch.Tensor:
        # return indices of most similar set per row
        return self._min[1]

    def sample(self) -> torch.Tensor:
        """Returns an index Tensor where the where the first row corresponds to
        the anchor set indices and the second row corresponds to the indices of
        the positive set.
        """
        pos_sample_idx = self._positive_sampling()
        anchor_idx = torch.arange(self._emd.size(0), device=self._device)

        # [2, b] => (anchor_idx, positive _idx)
        samples = torch.vstack((anchor_idx, pos_sample_idx))
        return samples


# samples positive and negative samples from the same space
# wsset pointswap sampling samples positive samples from EMD space (ie these are constant)
# but samples negative samples from embedding space (ie changes as model updates)
class TripletSampler(PositiveSampler):
    _SMALL_RANGE_DISTANCES = {"cosine", "angular"}

    def __init__(
        self,
        emd: torch.Tensor,
        disttype: DISTANCE_TYPES = "euclidean",
        scale: float = 7.0,
    ):
        super().__init__(emd)
        self._scale = scale
        self._disttype = disttype

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

    def sample(self) -> TripletSample:
        return self.triplet_sampling()


def negative_sampling(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    *,
    pos_idx: Optional[torch.Tensor] = None,
    scale: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:

    _SENTINEL = float("inf")
    if Y is None:
        # this chooses the neg sample from the real dataset
        # otherwise, Y is the augmented data, and this will choose from Y
        Y = X
        if pos_idx is None:
            raise ValueError("pos_idx must be supplied if a second tensor Y is not")
    else:
        # pos_idx is just aligned already in the case of sampling from point swapped augmented data
        pos_idx = torch.arange(Y.size(0))

    # in case of sampling from real dataset, diag should be 0 due to self-comparison
    # so don't allow self to be chosen as neg sample
    # for pnt swap, generally diagonal should be the smallest, so chose another
    dist = torch.cdist(X, Y)
    std = dist.std()
    # this is not required since self-comp will always be less than or equal to pos_dist
    # dist[dist <= torch.diag(dist).unsqueeze(1)] = _SENTINEL
    idx = torch.arange(dist.size(0))

    # don't allow points closer than the pos sample be chosen
    # this also has the effect of removing self-comparisons
    pos_dists = dist[idx, pos_idx].unsqueeze(1)
    dist[dist <= pos_dists] = _SENTINEL

    # randomly sample from remaining choices
    choice_idx, choice_opt = torch.where(dist != _SENTINEL)
    choices = defaultdict(list)
    for i, j in zip(choice_idx, choice_opt):
        i = int(i)
        j = int(j)
        choices[i].append(j)

    # TODO: could get out of order?
    neg_idx = torch.tensor([random.choice(opts) for opts in choices.values()])
    # calc negative sample weight
    neg_dists = dist[idx, neg_idx]
    denom = 2 * (scale * std) ** 2
    neg_weight = torch.exp(-neg_dists / denom)

    return neg_idx, neg_weight


class PointSwapSampler:
    _SENTINEL = -1

    def __init__(
        self,
        emd: torch.Tensor,
        batch: torch.Tensor,
        flow: FlowDict,
        row_mask: torch.Tensor,
        sample_rate: float,
        distfunc: DistFuncSignature = euclidean_distance,
    ):
        self._batch = batch
        self._flow = flow
        self._sample_rate = sample_rate
        self._distfunc = distfunc
        # indicates which rows are real ptns since batch is row-padded
        self._row_mask = row_mask
        self._anchor_idx = set(range(self._row_mask.size(-1)))
        self._rwmdistance = SetDistance(distfunc)
        self._positive_sampler = PositiveSampler(emd=emd)
        self._positive_sample = self._positive_sampler.sample()
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

    def point_swap_sampling(self) -> torch.Tensor:
        augmented_samples = [self._point_swap(i, j) for i, j in self._positive_idx]
        X_aug = torch.stack(augmented_samples)
        return X_aug

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._positive_sample, self.point_swap_sampling()


class PointSwapTripletSampler(PointSwapSampler):
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


PrecomputedSampling = dict[str, dict[int, torch.Tensor]]


class PrecomputeSampler:
    def __init__(
        self,
        data_file: Path,
        batch_size: int,
        dataloader: DataLoader,
        sample_rate: float = 0.5,
        distfunc: DistFuncSignature = euclidean_distance,
        device: Literal["cpu", "gpu", "auto"] = "cpu",
    ):
        self.file = self.get_precomputed_sampling_filename(
            data_file=data_file,
            batch_size=batch_size,
            sample_rate=sample_rate,
        )
        self.dataloader = dataloader
        self.sample_rate = sample_rate
        self.distfunc = distfunc
        self.device = torch.device(
            "cuda" if device == "gpu" else "cpu" if device == "auto" else device
        )
        self._get_external_file()
        self.precomputed_sampling = self._precompute_point_swap_sampling()

    def _precompute(self) -> PrecomputedSampling:
        RWMDDistance = SetDistance(distfunc=self.distfunc)
        positive_sample_indices: dict[int, torch.Tensor] = dict()
        aug_sample_data: dict[int, torch.Tensor] = dict()
        for batch_idx, batch in self.dataloader:
            row_mask = compute_row_mask(batch)
            rwmd, flow = RWMDDistance.fit_transform(batch)
            sampler = PointSwapSampler(
                emd=rwmd,
                batch=batch,
                flow=flow,
                row_mask=row_mask,
                sample_rate=self.sample_rate,
            )

            pos_sample, aug_sample = sampler.sample()
            pos_sample = pos_sample.to(device=self.device)
            aug_sample = aug_sample.to(device=self.device)

            positive_sample_indices[batch_idx] = pos_sample
            aug_sample_data[batch_idx] = aug_sample

        samples = {"positive": positive_sample_indices, "aug": aug_sample_data}
        return samples

    def _get_external_file(self):
        # check in cwd
        try:
            # specifically for CHTC to check if file exists there
            datadir = Path(os.environ["STAGING"]).joinpath(os.environ["USER"])
        except KeyError:
            datadir = Path.cwd()

        self._ext_file = datadir.joinpath(self.file.name)

    def exists(self) -> bool:
        return self._ext_file.exists() or self.file.exists()

    def _precompute_point_swap_sampling(self) -> PrecomputedSampling:
        if not self.exists():
            # check both STAGING and cwd
            print(f"Calculating precomputed point swap sampling to: {self.file}")
            return self._precompute()
        elif self.file.exists():
            # for subsequent runs, if file copied from CHTC /staging
            # don't try to recopy, just use the local file
            # it will be loaded later
            pass
        elif self._ext_file.exists():
            # file located in CHTC /staging
            # copy file over from staging to cwd
            shutil.copyfile(self._ext_file, Path.cwd().joinpath(self.file))

        # STAGING file copied over already, so this works in either case
        print(f"Loading {self.file}")
        return self.load()

    def precompute_point_swap_sampling(self) -> PrecomputedSampling:
        return self._precompute_point_swap_sampling()

    def save(self):
        with self.file.open("wb") as fp:
            torch.save(self.precomputed_sampling, fp)

    def load(self, device: Optional[torch.device] = None) -> PrecomputedSampling:
        return self.load_precomputed_sampling(self.file, device)

    def to(self, device: Optional[torch.device] = None):
        _move_precomputed_sampling_to_device(self.precomputed_sampling, device)

    @staticmethod
    def load_precomputed_sampling(
        file: Path, device: Optional[torch.device] = None
    ) -> PrecomputedSampling:
        precomputed_sampling = torch.load(file)
        _move_precomputed_sampling_to_device(precomputed_sampling, device)
        return precomputed_sampling

    @staticmethod
    def get_precomputed_sampling_filename(
        data_file: Path, batch_size: int, sample_rate: float
    ) -> Path:
        _output = [
            data_file.stem,
            "precomputed-sampling",
            f"batchsize-{batch_size}",
            f"sample-rate-{sample_rate}",
        ]
        output = Path.cwd().joinpath(f"{'_'.join(_output)}.pt")
        return output


def _move_precomputed_sampling_to_device(
    precomputed_sampling: PrecomputedSampling,
    device: Optional[torch.device] = torch.device("cpu"),
):
    if device is not None:
        for sampletype, sampledata in precomputed_sampling.items():
            for batch_id, batch_data in sampledata.items():
                precomputed_sampling[sampletype][batch_id] = batch_data.to(
                    device=device
                )


def get_precomputed_sampling_filename_from_args(args: Args) -> Path:
    return PrecomputeSampler.get_precomputed_sampling_filename(
        data_file=args.data["data_file"],
        batch_size=args.data["batch_size"],
        sample_rate=args.model["sample_rate"],
    )

from __future__ import annotations

import os
import shutil
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


PrecomputedSampling = dict[str, dict[str, dict[int, torch.Tensor]]]


class PrecomputeSampler:
    def __init__(
        self,
        data_file: Path,
        batch_size: int,
        dataloader: DataLoader,
        sample_rate: float = 0.5,
        scale: float = 7.0,
        distfunc: DistFuncSignature = euclidean_distance,
        device: Literal["cpu", "gpu", "auto"] = "cpu",
    ):
        self.file = self.get_precomputed_sampling_filename(
            data_file=data_file,
            batch_size=batch_size,
            sample_rate=sample_rate,
            sample_scale=scale,
        )
        self.dataloader = dataloader
        self.sample_rate = sample_rate
        self.sample_scale = scale
        self.distfunc = distfunc
        self.device = torch.device(
            "cuda" if device == "gpu" else "cpu" if device == "auto" else device
        )
        self._get_external_file()
        self.precomputed_sampling = self._precompute_point_swap_sampling()

    def _precompute(self) -> PrecomputedSampling:
        RWMDDistance = SetDistance(distfunc=self.distfunc)
        triplet_sample_indices: dict[int, torch.Tensor] = dict()
        triplet_sample_weights: dict[int, torch.Tensor] = dict()
        aug_sample_data: dict[int, torch.Tensor] = dict()
        aug_sample_neg_indices: dict[int, torch.Tensor] = dict()
        aug_sample_weights: dict[int, torch.Tensor] = dict()
        for batch_idx, batch in self.dataloader:
            row_mask = compute_row_mask(batch)
            rwmd, flow = RWMDDistance.fit_transform(batch)
            sampler = PointSwapSampler(
                emd=rwmd,
                batch=batch,
                flow=flow,
                row_mask=row_mask,
                sample_rate=self.sample_rate,
                scale=self.sample_scale,
                distfunc=self.distfunc,
            )
            triplet_sample, aug_sample = sampler.sample()
            triplet_sample = triplet_sample.to(device=self.device)
            aug_sample = aug_sample.to(device=self.device)

            triplet_sample_indices[batch_idx] = triplet_sample.idx
            triplet_sample_weights[batch_idx] = triplet_sample.weights
            aug_sample_data[batch_idx] = aug_sample.data
            aug_sample_neg_indices[batch_idx] = aug_sample.negative_idx
            aug_sample_weights[batch_idx] = aug_sample.weights

        samples = {
            "triplet": {
                "indices": triplet_sample_indices,
                "weights": triplet_sample_weights,
            },
            "aug": {
                "data": aug_sample_data,
                "weights": aug_sample_weights,
                "negative_indices": aug_sample_neg_indices,
            },
        }
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
        data_file: Path, batch_size: int, sample_rate: float, sample_scale: float
    ) -> Path:
        _output = [
            data_file.stem,
            "precomputed-sampling",
            f"batchsize-{batch_size}",
            f"sample-rate-{sample_rate}",
            f"sample-scale-{sample_scale}",
        ]
        output = Path.cwd().joinpath(f"{'_'.join(_output)}.pt")
        return output


def _move_precomputed_sampling_to_device(
    precomputed_sampling: PrecomputedSampling,
    device: Optional[torch.device] = torch.device("cpu"),
):
    if device is not None:
        for sampletype, sampledata in precomputed_sampling.items():
            for valuename, data in sampledata.items():
                for batch_id, batch_data in data.items():
                    precomputed_sampling[sampletype][valuename][
                        batch_id
                    ] = batch_data.to(device=device)


def get_precomputed_sampling_filename_from_args(args: Args) -> Path:
    return PrecomputeSampler.get_precomputed_sampling_filename(
        data_file=args.data["data_file"],
        batch_size=args.data["batch_size"],
        sample_rate=args.model["sample_rate"],
        sample_scale=args.model["sample_scale"],
    )

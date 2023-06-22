from __future__ import annotations

from typing import cast, Iterator, TypeVar, Generic

import numpy as np
import torch
from numpy.typing import NDArray

Int64Array = NDArray[np.int64]
LongArray = TypeVar("LongArray", Int64Array, torch.Tensor)


class ImbalancedGroupKFold(Generic[LongArray]):
    def __init__(self, groups: LongArray) -> None:
        if isinstance(groups, torch.Tensor):
            self.index_fn = torch.arange
        elif isinstance(groups, np.ndarray):
            self.index_fn = np.arange

        # send outputs as the same tensor type as input
        self.X_idx = self.get_X_index(groups)

        # but internally do everything as numpy arrays
        self.uniq_groups: Int64Array
        self.groups: Int64Array
        # self.groups maps actual group labels to consecutively increasing,
        # ie [0, 1, ..., n-1, n]
        self.uniq_groups, self.groups = np.unique(groups, return_inverse=True)
        self.group_counts = np.bincount(self.groups)
        self._sort_groups()

        # don't have a fold for the most frequent group
        self.n_folds = self.uniq_groups.shape[0] - 1
        self.largest_group_id = np.argmax(self.group_counts)

    def get_X_index(self, groups: LongArray) -> LongArray:
        n = groups.shape[0]
        return cast(LongArray, self.index_fn(n))

    def _sort_groups(self):
        # reverse sort, ie 0th element is count of most frequent group
        sort_idx = np.argsort(self.group_counts)[::-1]
        self.uniq_groups = self.uniq_groups[sort_idx]
        self.group_counts = self.group_counts[sort_idx]

    def __len__(self) -> int:
        return self.n_folds

    def split(self) -> Iterator[tuple[LongArray, LongArray]]:
        for fold_idx, group_id in enumerate(self.uniq_groups):
            if fold_idx == 0:
                # the first group will be the most frequent and
                # we don't want a fold where teh most frequent group
                # is the validation set
                continue

            val_mask = np.where(self.groups == group_id, True, False)
            train_mask = ~val_mask

            train_idx = self.X_idx[train_mask]
            val_idx = self.X_idx[val_mask]

            yield train_idx, val_idx

from typing import Iterable, Iterator, Optional

import torch
from torch import nn


class LayerDropModuleList(nn.ModuleList):
    def __init__(
        self, probability: float = 0.0, modules: Optional[Iterable[nn.Module]] = None
    ):
        super().__init__(modules)
        self.p = probability

    def __iter__(self) -> Iterator[nn.Module]:
        dropout_prob = torch.rand(len(self))

        for module_idx, module in enumerate(super().__iter__()):
            if not self.training or dropout_prob[module_idx] > self.p:
                yield module

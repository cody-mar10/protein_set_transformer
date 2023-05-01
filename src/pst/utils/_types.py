from __future__ import annotations

from typing import Callable
import torch

FlowDict = dict[tuple[int, int], torch.Tensor]
DistFuncSignature = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BatchType = tuple[torch.Tensor, torch.Tensor]

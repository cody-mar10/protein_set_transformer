from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AnnealingOpts = Literal["linear", "cos"]


class ExperimentArgs(BaseModel):
    name: str = Field(
        "exp0", description="experiment name in logging directory during training"
    )
    patience: int = Field(
        5, description="early stopping patience, see lightning description", gt=0
    )
    save_top_k: int = Field(3, description="save the best k models", gt=0)
    ### SWA args
    swa: bool = Field(False, description="use stochastic weight averaging")
    swa_epoch_start: int | float = Field(
        0.8,
        description=(
            "when to start SWA if enabled. Integers are an epoch number, "
            "while floats in range (0.0, 1.0) are a fraction of max epochs. "
        ),
    )
    annealing_epochs: int = Field(
        10,
        description="number of epochs to anneal learning rate during swa if enabled",
        gt=0,
    )
    annealing_strategy: AnnealingOpts = Field(
        "linear", description="annealing strategy using during swa if enabled"
    )

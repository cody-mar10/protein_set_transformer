from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .utils import register_model

AnnealingOpts = Literal["linear", "cos"]
_NAME = "experiment"


@register_model(_NAME)
class ExperimentArgs(BaseModel):
    config: Optional[Path] = Field(
        None,
        description="config toml file to specify which hyperparameters to tune with optuna",  # noqa: E501
    )
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
    ### TUNING args
    tune: bool = Field(
        True,
        description="whether to use hyperparameter sampling/tuning during training",
    )
    n_trials: int = Field(100, description="number of tuning trials to run", gt=0)
    prune: bool = Field(
        True, description="whether to allow pruning of unpromising trials"
    )
    parallel: bool = Field(
        False,
        description="whether to run tuning trials in parallel if multiple GPUs are available",  # noqa: E501
    )

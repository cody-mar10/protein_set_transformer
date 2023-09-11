from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class TuningArgs(BaseModel):
    config: Optional[Path] = Field(
        None,
        description="config toml file to specify which hyperparameters to tune with optuna",  # noqa: E501
    )
    n_trials: int = Field(1, description="number of tuning trials to run", gt=0)
    prune: bool = Field(
        True, description="whether to allow pruning of unpromising trials"
    )
    parallel: bool = Field(
        False,
        description="NOT IMPL: whether to run tuning trials in parallel if multiple GPUs are available",  # noqa: E501
    )
    tuning_dir: Optional[Path] = Field(
        None,
        description="tuning directory path that stores optuna SQL history databases",
    )
    pruning_warmup_steps: int = Field(
        3, description="number of epochs before pruning can start"
    )
    pruning_warmup_trials: int = Field(
        1, description="number of trials allowed to complete before pruning can start"
    )

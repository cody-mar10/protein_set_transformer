from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jsonargparse.typing import NonNegativeInt, PositiveInt

from pst.utils.dataclass_utils import DataclassValidatorMixin, validated_field


@dataclass
class TuningArgs(DataclassValidatorMixin):
    """TUNING

    Hyperparameter tuning configuration.
    """

    # TODO: this makes most sense here.... 
    config: Path
    """config toml file to specify which hyperparameters to tune with optuna"""

    n_trials: int = validated_field(1, validator=PositiveInt)
    """number of tuning trials to run"""

    prune: bool = True
    """whether to allow pruning of unpromising trials"""

    parallel: bool = False
    """NOT IMPL: whether to run tuning trials in parallel if multiple GPUs are available"""

    tuning_dir: Optional[Path] = None
    """tuning directory path that stores optuna SQL history databases"""

    pruning_warmup_steps: int = validated_field(3, validator=PositiveInt)
    """number of epochs before pruning can start"""

    pruning_warmup_trials: int = validated_field(1, validator=NonNegativeInt)
    """number of trials allowed to complete before pruning can start"""

from pathlib import Path
from typing import Optional

from attrs import define, field

from pst.utils.attrs.dataclass_utils import AttrsDataclassUtilitiesMixin
from pst.utils.attrs.validators import (
    file_exists,
    non_negative_int,
    optionally_existing_file,
    positive_int,
)


@define
class TuningArgs(AttrsDataclassUtilitiesMixin):
    """TUNING

    Hyperparameter tuning configuration.
    """

    config: Path = field(validator=file_exists)
    """config toml file to specify which hyperparameters to tune with optuna"""

    n_trials: int = field(default=1, validator=positive_int)
    """number of tuning trials to run"""

    prune: bool = True
    """whether to allow pruning of unpromising trials"""

    parallel: bool = False
    """NOT IMPL: whether to run tuning trials in parallel if multiple GPUs are available"""

    tuning_dir: Optional[Path] = field(default=None, validator=optionally_existing_file)
    """tuning directory path that stores optuna SQL history databases"""

    pruning_warmup_steps: int = field(default=3, validator=positive_int)
    """number of epochs before pruning can start"""

    pruning_warmup_trials: int = field(default=1, validator=non_negative_int)
    """number of trials allowed to complete before pruning can start"""

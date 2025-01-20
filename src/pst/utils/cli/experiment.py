from pathlib import Path
from typing import Optional

from attrs import define, field

from pst.utils.attrs.dataclass_utils import AttrsDataclassUtilitiesMixin
from pst.utils.attrs.validators import optionally_existing_file


@define
class ExperimentArgs(AttrsDataclassUtilitiesMixin):
    """EXPERIMENT"""

    name: str = "exp0"
    """experiment name in logging directory during training"""

    best_trial: Optional[Path] = field(default=None, validator=optionally_existing_file)
    """load hyperparameters from a previous tuning run. Acceptable inputs include:
    - (RECOMMENDED) an optuna sqlite database that holds the entire history of a tuning study
    - a json file that holds the best hyperparameters from a tuning study produced by `lightning_cv`
        - Otherwise, this can be any json file that has a key `hparams` whose value is a flat dictionary
        of hyperparameter names and sampled values
    - a yaml file that holds the best hyperparameters from a tuning study. Note: the yaml file
    can also be a config file produced by `jsonargparse` that holds the best hyperparameters.
    NOTE: this is just a convenience feature to load all hyperparameters from a previous tuning run,
    but they can also be entered individually at the command line.
    """

    config: Optional[Path] = field(default=None, validator=optionally_existing_file)
    """Tuning trial hyperparameter config file that was used to define the sampling space.
    This is the same file that should've been passed to `--tuning.config` during hyperparameter
    tuning. This is needed if any hyperparameters were sampled from a categorical distribution
    whose values get mapped to other inputs. Ex: Sampling a model single `complexity` 
    hyperparameter that adjusts the number of layers, attention heads, embedding dim, etc.
    """

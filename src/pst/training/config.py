from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Generic, Literal, Optional, Sequence, TypeVar

import optuna  # noqa
from pydantic import BaseModel, Field, ValidationError

from pst.utils.versions import PYTHON_VERSION_GTE_3_11

if PYTHON_VERSION_GTE_3_11:
    import tomllib as tomli  # type: ignore
else:
    import tomli

_T = TypeVar("_T")
_U = TypeVar("_U")


class TunableInt(BaseModel):
    suggest: Literal["int"]
    low: int
    high: int
    step: int = 1


class TunableFloat(BaseModel):
    suggest: Literal["float"]
    low: float
    high: float
    log: bool = False


class TunableCategorical(BaseModel, Generic[_T, _U]):
    suggest: Literal["categorical"]
    choices: Sequence[_T]
    map: Optional[dict[_T, dict[str, _U]]]


TunableType = Annotated[
    TunableInt | TunableFloat | TunableCategorical[_T, _U],
    Field(discriminator="suggest"),
]
ConfigType = dict[str, TunableType]


class Config(BaseModel):
    model: ConfigType
    optim: ConfigType
    loss: ConfigType
    augmentation: ConfigType
    data: ConfigType
    experiment: ConfigType


# ex format: {"model": {"hparam": {**opts}}}
def load_config(file: Path) -> dict[str, dict[str, dict[str, Any]]]:
    with file.open("rb") as fp:
        config = tomli.load(fp)

    try:
        validated_config = Config.validate(config).dict()
    except ValidationError as e:
        print(str(e))
        raise e

    return validated_config


_DEFAULT_CONFIG = {
    "model": {
        "complexity": {
            "suggest": "categorical",
            "choices": ["small", "medium", "large", "xl", "xxl"],
            "map": {
                "small": {"num_heads": 4, "n_enc_layers": 5},
                "medium": {"num_heads": 8, "n_enc_layers": 10},
                "large": {"num_heads": 16, "n_enc_layers": 15},
                "xl": {"num_heads": 32, "n_enc_layers": 20},
                "xxl": {"num_heads": 32, "n_enc_layers": 30},
            },
        },
        "multiplier": {"suggest": "float", "low": 0.5, "high": 10.0, "log": True},
        "dropout": {"suggest": "float", "low": 0.0, "high": 0.5, "log": False},
    },
    "optim": {
        "lr": {"suggest": "float", "low": 0.001, "high": 0.1, "log": True},
        "weight_decay": {"suggest": "float", "low": 0.001, "high": 0.1, "log": True},
    },
    "data": {
        "batch_size": {"suggest": "categorical", "choices": [16, 32, 64]},
        "chunk_size": {
            "suggest": "int",
            "low": 15,
            "high": 50,
            "log": False,
            "step": 5,
        },
    },
    "loss": {"margin": {"suggest": "float", "low": 0.1, "high": 10.0, "log": True}},
    "experiment": {
        "swa_epoch_start": {"suggest": "float", "low": 0.5, "high": 0.8, "log": False},
        "annealing_epochs": {
            "suggest": "int",
            "low": 10,
            "high": 100,
            "log": False,
            "step": 1,
        },
        "annealing_strategy": {
            "suggest": "categorical",
            "choices": ["linear", "cosine"],
        },
    },
    "augmentation": {
        "sample_scale": {"suggest": "float", "low": 1.0, "high": 15.0, "log": False},
        "sample_rate": {"suggest": "float", "low": 0.25, "high": 0.75, "log": False},
    },
}

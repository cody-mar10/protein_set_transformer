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

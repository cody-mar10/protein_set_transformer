import json
import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, TypedDict, TypeVar, cast

import optuna
import yaml
from lightning_cv.tuning.config import MappingT, get_mappings

from pst.data.config import DataConfig
from pst.nn.config import BaseModelConfig
from pst.training.tuning.manager import StudyManager
from pst.typing import KwargType

ModelConfigT = TypeVar("ModelConfigT", bound=BaseModelConfig)
DataConfigT = TypeVar("DataConfigT", bound=DataConfig)

_ALLOWED_CONFIGS = Literal["model", "data"]


class History(NamedTuple):
    model: KwargType
    data: KwargType


def load_history(file: Path) -> dict[str, Any]:
    if file.suffix == ".json":
        with open(file) as fp:
            history = json.load(fp)["hparams"]
    elif file.suffix == ".db":
        url = StudyManager.create_url(file)
        study = optuna.load_study(study_name=StudyManager.STUDY_NAME, storage=url)
        history = study.best_params
    elif file.suffix in {".yaml", ".yml"}:
        # this should be handled at cli by jsonargparse
        # it should be something more like pst train --config CONFIG.yml
        logging.warning(
            "YAML files are likely produced by jsonargparse, which can be read directly "
            "from the command line. It is recommended to pass this file to --config instead if "
            "this file was created by the cli feature of jsonargparse --print_config. If not, "
            "then ignore this message."
        )

        with open(file) as fp:
            maybe_history = yaml.load(fp, Loader=yaml.CLoader)

        ## this checks jsonargparse format
        # need to post process
        # if "model" in maybe_history:
        #     maybe_history = maybe_history["model"]

        # if "init_args" in maybe_history:
        #     maybe_history = maybe_history["init_args"]["config"]

        history = maybe_history
    else:
        raise ValueError(f"unknown history file type: {file}")

    return history


def _prepare_history(history: KwargType) -> KwargType:
    not_flat = any(isinstance(value, dict) for value in history.values())
    if not_flat:
        # likely yaml?
        model_history = history.get("model", {}).get("init_args", {}).get("config", {})
        data_history = history.get("data", {})
        history = {**model_history, **data_history}

    # else: it likely came from lcv json file or optuna history
    return history


def _update_mappings(history: KwargType, mappings: MappingT) -> KwargType:
    for mapped_key, mapping_values in mappings.items():
        chosen_value = history.pop(mapped_key)
        real_values = mapping_values[chosen_value]
        history.update(real_values)
    return history


def _split_history(
    model_config: BaseModelConfig,
    data_config: DataConfig,
    history: KwargType,
) -> History:
    model_fields = set(model_config.fields())
    data_fields = set(data_config.fields())

    model_hparams = {}
    data_hparams = {}

    for key, value in history.items():
        if key in model_fields:
            model_hparams[key] = value
        elif key in data_fields:
            data_hparams[key] = value
        else:
            raise ValueError(
                f"Key {key} not found in model's config {model_fields} or data's config {data_fields}"
            )

    return History(model=model_hparams, data=data_hparams)


def _update_from_history(
    history: History, config: KwargType, config_type: _ALLOWED_CONFIGS
):
    """Update config from history inplace"""
    hparams: KwargType = getattr(history, config_type)

    stack = [config]
    while stack:
        current = stack.pop()
        for key, value in current.items():
            if isinstance(value, dict):
                stack.append(value)
            else:
                new_value = hparams.get(key, value)
                current[key] = new_value


class _ConfigInfo(TypedDict):
    dict: KwargType
    type: type[BaseModelConfig] | type[DataConfig]


def update_from_history(
    history: History, model_config: ModelConfigT, data_config: DataConfigT
) -> tuple[ModelConfigT, DataConfigT]:
    config: dict[_ALLOWED_CONFIGS, _ConfigInfo] = {
        "model": {
            "dict": model_config.to_dict(),
            "type": type(model_config),
        },
        "data": {
            "dict": data_config.to_dict(),
            "type": type(data_config),
        },
    }

    for config_type, config_info in config.items():
        _update_from_history(history, config_info["dict"], config_type)

    new_model_config = cast(
        ModelConfigT, config["model"]["type"].from_dict(config["model"]["dict"])
    )
    new_data_config = cast(
        DataConfigT, config["data"]["type"].from_dict(config["data"]["dict"])
    )

    return new_model_config, new_data_config


def update_config_from_history(
    model_config: ModelConfigT,
    data_config: DataConfigT,
    history_file: Path,
    tuning_config_def: Optional[Path] = None,
) -> tuple[ModelConfigT, DataConfigT]:
    history_dict = load_history(history_file)
    history_dict = _prepare_history(history_dict)
    if tuning_config_def is not None:
        mappings = get_mappings(tuning_config_def)
        history_dict = _update_mappings(history_dict, mappings)

    history = _split_history(model_config, data_config, history_dict)

    model_config, data_config = update_from_history(history, model_config, data_config)
    return model_config, data_config

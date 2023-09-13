from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, TypeVar

import optuna
from lightning_cv.tuning.config import MappingT, get_mappings
from pydantic import BaseModel

from pst.training.tuning.manager import StudyManager
from pst.utils.cli.modes import TrainingMode

ModelT = TypeVar("ModelT", bound=BaseModel)
ModeT = TypeVar("ModeT", bound=TrainingMode)


def load_history(file: Path) -> dict[str, Any]:
    if file.suffix == ".json":
        with open(file) as fp:
            history = json.load(fp)["hparams"]
    elif file.suffix == ".db":
        url = StudyManager.create_url(file)
        study = optuna.load_study(study_name=StudyManager.STUDY_NAME, storage=url)
        history = study.best_params
    else:
        raise ValueError(f"unknown history file type: {file}")

    return history


def _update_from_history(history: dict[str, Any], config: dict[str, Any]):
    for key, value in config.items():
        if isinstance(value, dict):
            _update_from_history(history, value)
        else:
            new_value = history.get(key, value)
            config[key] = new_value


def update_from_history(
    history: dict[str, Any], config: ModelT, mapping: Optional[MappingT] = None
) -> ModelT:
    config_ser = config.model_dump()

    if mapping is not None:
        for mapped_key, mapping_values in mapping.items():
            chosen_value = history.pop(mapped_key)
            real_values = mapping_values[chosen_value]
            history.update(real_values)

    _update_from_history(history, config_ser)
    return config.model_validate(config_ser)


def update_config_from_history(args: ModeT) -> ModeT:
    history_file = args.experiment.from_study or args.experiment.from_json
    if history_file is not None:
        history_dict = load_history(history_file)

        if args.experiment.config is not None:
            mappings = get_mappings(args.experiment.config)
        else:
            mappings = None

        args = update_from_history(history_dict, args, mappings)
    return args

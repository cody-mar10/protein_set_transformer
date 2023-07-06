from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class Register(Generic[ModelT]):
    def __init__(self):
        self._defaults: dict[str, dict[str, Any]] = dict()
        self._fields: dict[str, list[str]] = dict()
        self._models: dict[str, ModelT] = dict()

    def _get_fields(self, model: ModelT) -> list[str]:
        return list(model.__fields__.keys())

    def register(self, key: str):
        def _register(cls: ModelT) -> ModelT:
            self._models[key] = cls
            self._fields[key] = self._get_fields(cls)
            instance = cls.construct()
            self._defaults[key] = instance.dict()
            return cls

        return _register

    def get_fields(self, key: Optional[str] = None):
        if key is None:
            return self._fields
        return self._fields[key]

    def get_defaults(self, key: Optional[str] = None):
        if key is None:
            return self._defaults
        return self._defaults[key]

    def get_models(self, key: Optional[str] = None):
        if key is None:
            return self._models
        return self._models[key]

    def split_fields(self, model: ModelT) -> dict[str, dict[str, Any]]:
        config: dict[str, dict[str, Any]] = dict()

        for field_type, field_names in self._fields.items():
            # only include parameters that have been added to each submodel
            if all(hasattr(model, field_name) for field_name in field_names):
                config[field_type] = model.dict(include=set(field_names))
        return config


_REGISTRY = Register()

register_model = _REGISTRY.register

get_fields = _REGISTRY.get_fields
get_defaults = _REGISTRY.get_defaults
get_models = _REGISTRY.get_models

split_fields = _REGISTRY.split_fields

_NONEXISTENT_FILE = Path("__NONEXISTENT_FILE__")


_T = TypeVar("_T")


def _validate_and_call_predicate(
    x: str, val: _T, predicate: Optional[Callable[[_T], bool]] = None
):
    if predicate is not None:
        if not predicate(val):
            raise ValueError(
                f"Failed validation check when parsing CLI args: {x} -> {val}"
            )


def parse_int_or_float_arg(
    x: str,
    int_predicate: Optional[Callable[[int], bool]] = None,
    float_predicate: Optional[Callable[[float], bool]] = None,
) -> int | float:
    if x.isdigit():
        val = int(x)
        _validate_and_call_predicate(x, val, int_predicate)
        return val

    val = float(x)
    _validate_and_call_predicate(x, val, float_predicate)
    return val


def validate_proportion_range(
    x: float, left_inclusive: bool = True, right_inclusive: bool = True
) -> bool:
    low = 0.0
    high = 1.0
    if left_inclusive and right_inclusive:
        return low <= x <= high
    elif left_inclusive:
        return low <= x < high
    elif right_inclusive:
        return low < x <= high
    return low < x < high

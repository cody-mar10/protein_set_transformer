from __future__ import annotations

import argparse
from dataclasses import asdict as dc_asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

_ADDER_TYPE = Callable[[argparse.ArgumentParser], None]
_ARGPARSE_HANDLERS: list[_ADDER_TYPE] = list()
_DEFAULTS: dict[str, dict[str, Any]] = dict()


def register_defaults(defaults, key: str):
    _DEFAULTS[key] = dc_asdict(defaults)


def get_defaults(key: str) -> dict[str, Any]:
    return _DEFAULTS[key]


def register_handler(func: _ADDER_TYPE):
    _ARGPARSE_HANDLERS.append(func)


_NONEXISTENT_FILE = Path("__NONEXISTENT_FILE__")

_PARSER_TYPE = Callable[[argparse.Namespace], Any]


def asdict(fn: _PARSER_TYPE):
    """dataclasses.dataclass asdict decorator"""

    def wrap(args: argparse.Namespace):
        output = fn(args)
        if is_dataclass(output):
            return dc_asdict(output)
        raise ValueError(
            f"Input fn {fn.__name__} does not return a dataclasses.dataclass instance"
        )

    return wrap


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

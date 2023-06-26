from __future__ import annotations

import argparse
from dataclasses import asdict as dc_asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable

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

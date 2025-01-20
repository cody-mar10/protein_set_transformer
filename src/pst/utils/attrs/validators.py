"""Validators for `attrs.define` dataclass-like config objects."""

# all validators take 3 args ->
# [1] the formed instance of the dataclass obj
# [2] the attrs.Attribute obj
# [3] the value of the attribute

import os

from attrs import Attribute
from attrs.validators import and_, ge, gt, instance_of, le, lt, optional, or_

from pst.typing import FilePath

is_int = instance_of(int)
is_float = instance_of(float)

positive_int = and_(is_int, gt(0))
non_negative_int = and_(is_int, ge(0))
positive_float = and_(is_float, gt(0.0))
non_negative_float = and_(is_float, ge(0.0))
open_unit_interval = and_(is_float, gt(0.0), lt(1.0))
closed_unit_interval = and_(is_float, ge(0.0), le(1.0))


def equals(expected_value):
    return and_(ge(expected_value), le(expected_value))


optional_positive_int = or_(positive_int, equals(-1))


def file_exists(instance, attribute: Attribute, value: FilePath):
    if not os.path.exists(value):
        raise FileNotFoundError(f"{attribute.name} does not exist: {value}")


optionally_existing_file = optional(file_exists)

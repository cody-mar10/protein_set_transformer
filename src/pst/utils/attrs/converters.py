from enum import Enum
from functools import partial
from typing import Any, Callable, TypeVar, Union
from warnings import warn

from attrs import NOTHING, field

_E = TypeVar("_E", bound=Enum)


class EnumConverter:
    @staticmethod
    def convert(value: Union[_E, str, Any], enum_cls: type[_E]) -> _E:
        """Convert an enum member value or name to the enum member, or return the enum member
        itself.

        This is the recommended way to for user code to directly convert enum values to enum
        members.

        Args:
            value (Union[_E, str, Any]): possible value of the enum member
            enum_cls (type[_E]): the enum class to convert to

        Raises:
            ValueError: if the `value` is not a valid enum member or enum member value

        Returns:
            _E: an instance of the enum
        """

        if isinstance(value, enum_cls):
            return value

        for member in enum_cls.__members__.values():
            if value == member.name or value == member.value:
                return member

        raise ValueError(
            f"{value} is not a valid member of {enum_cls} and could not convert."
        )

    @staticmethod
    def value_converter(enum_cls: type[_E]) -> Callable[[Union[_E, str, Any]], _E]:
        """Get a callable that converts an enum member value or name to the enum member, or
        returns the enum member itself.

        Args:
            enum_cls (type[_E]): the enum class to convert to

        Returns:
            Callable[[Union[_E, str, Any]], _E]: a function that accepts a value and tries to
                return the corresponding enum member
        """
        return partial(EnumConverter.convert, enum_cls=enum_cls)


def enum_field(*, enum_cls: type[_E], default: _E = NOTHING, **kwargs):
    """Create an attrs field that auto converts a name or value to an enum
    member.

    Args:
        enum_cls (type[_E]): the enum class to convert to. This is required for
            type hints to properly work.
        default (_E, optional): the default value of the field
        **kwargs: All other arguments to the `attrs.field` function
    """
    passed_converter = kwargs.pop("converter", None) is not None
    if passed_converter:
        warn(
            "The `converter` argument is not allowed for `enum_field` and will be ignored."
        )

    # note that this is preferable to `attrs.Converter` which has weird type signatures
    # and would still require the enum_cls type hint
    return field(
        default=default, converter=EnumConverter.value_converter(enum_cls), **kwargs
    )

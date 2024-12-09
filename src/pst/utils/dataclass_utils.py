import copy
import inspect
import re
from dataclasses import dataclass, field, fields, is_dataclass, Field
from enum import Enum
from io import StringIO
from typing import Callable, Iterator, Optional, Set, TypeVar

import yaml
from dataclass_wizard import fromdict

from pst.typing import KwargType

_T = TypeVar("_T")
_E = TypeVar("_E", bound=Enum)


def _asdict(model):
    """Convert `dataclasses.dataclass` to a dictionary. This is a separate implementation
    of `dataclasses.asdict` that does NOT deepcopy data.

    Args:
        model `dataclasses.dataclass` instance

    Returns:
        KwargType: dictionary representation of the dataclass with all str keys
    """
    ser: KwargType = {}

    stack: list = [(ser, model)]
    while stack:
        parent, curr_model = stack.pop()
        for f in fields(curr_model):
            value = getattr(curr_model, f.name)
            if is_dataclass(value):
                parent[f.name] = {}
                stack.append((parent[f.name], value))
            else:
                parent[f.name] = value

    return ser


def model_dump(
    model: object,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> KwargType:
    """Convert `dataclasses.dataclass` to a dictionary. Optionally include or exclude fields
    ONLY from the top-most level. Only `include` OR `exclude` can be used, not both.

    Args:
        model `dataclasses.dataclass` instance
        include (Optional[Set[str]], optional): set of top-level keys to include.
            Defaults to None, meaning include ALL.
        exclude (Optional[Set[str]], optional): set of top-level keys to exclude.
            Defaults to None, meaning exclude NOTHING.

    Raises:
        ValueError: if model is not a python built-in `dataclasses.dataclass` instance

    Returns:
        KwargType: dictionary representation of the dataclass with all str keys
    """
    if not is_dataclass(model):
        raise ValueError(f"model must be a dataclass got {type(model)}")

    if include and exclude:
        raise ValueError("Only include OR exclude can be used, not both")

    all_ser = _asdict(model)

    if include is not None:
        return {k: v for k, v in all_ser.items() if k in include}

    if exclude is not None:
        return {k: v for k, v in all_ser.items() if k not in exclude}

    return all_ser


def _convert_str_to_enum(value: str | _E, enum_type: type[_E]) -> _E:
    if isinstance(value, str) and not isinstance(
        value, Enum
    ):  # special case for StrEnum
        return enum_type[value]
    return value  # type: ignore


def validate(model):
    """Validate a dataclass instance that has field validators.

    This is basically like pydantic, but for dataclasses.

    This will also convert `str` name values to `Enum` instances if the field is an `Enum`.

    Args:
        model: dataclass instance

    Raises:
        ValueError: validation fails
    """
    if not is_dataclass(model):
        return

    for f in fields(model):
        validator = f.metadata.get("validator", None)
        value = getattr(model, f.name)

        if isinstance(f.type, type) and issubclass(f.type, Enum):
            try:
                value = _convert_str_to_enum(value, f.type)
            except KeyError as e:
                raise ValueError(
                    f"Validation failed for field `{f.name}` with value {value}: {e}"
                ) from e
            else:
                # replace the value with the Enum instance
                setattr(model, f.name, value)

        if validator is not None:
            try:
                validator(value)
            except ValueError as e:
                raise ValueError(
                    f"Validation failed for field `{f.name}` with value {value}: {e}"
                ) from e


class DocstringInheritorMetaclass(type):
    _GOOGLE_DOCSTRING_PATTERN = re.compile(r"(.*?)(?:\s+)?(?:\(.*?\))?:\s+?(.*)?$")
    # examples:
    # a (int): description
    # a: description

    _INDENT = 4

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        clsattrs: KwargType,
        *,
        attr_name: str = "Attributes",
    ):
        # cls refers to this metaclass
        # see we need to only use name, bases, attrs

        curr_doc = inspect.cleandoc(clsattrs.get("__doc__", ""))

        curr_attrs = cls._parse_docstring_attrs(
            curr_doc, attr_name, cls._GOOGLE_DOCSTRING_PATTERN
        )

        if bases:
            # don't need to go through all parents since if we have linear inheritance
            # then each previous one will get patched already
            # all_bases = cls._get_bases(bases=bases)
            all_bases = bases

            base_attrs_order: list[dict[str, str]] = []
            for base in all_bases:
                base_doc = inspect.getdoc(base)
                if base_doc is not None:
                    base_attr = cls._parse_docstring_attrs(
                        base_doc, attr_name, cls._GOOGLE_DOCSTRING_PATTERN
                    )

                    base_attrs_order.append(base_attr)
            base_attrs = {}
            for base in reversed(base_attrs_order):
                base_attrs.update(base)

            all_attrs = base_attrs | curr_attrs
            clsattrs["__doc__"] = cls._combine_docstrings(
                name, curr_doc, all_attrs, attr_name
            )
        return super().__new__(cls, name, bases, clsattrs)

    @staticmethod
    def _get_bases(*, cls: Optional[type] = None, bases: Optional[tuple[type, ...]] = None) -> list[type]:  # type: ignore
        if cls is None and bases is None:
            raise ValueError("Either cls or bases must be provided")
        elif cls is not None and bases is None:
            bases = cls.__bases__
        elif cls is None and bases is not None:
            pass
        else:
            raise ValueError("Only one of cls or bases should be provided")

        all_bases: list[type] = []
        for base in bases:
            if base is object or base in all_bases:
                continue

            all_bases.append(base)
            all_bases.extend(
                DocstringInheritorMetaclass._get_bases(bases=base.__bases__)
            )

        return all_bases

    @staticmethod
    def _search_until_attr(docstring_lines: list[str], attr_name: str) -> int:
        """Search docstring until the attribute name header is found.

        Args:
            docstring (list[str]): Google-style docstring split into lines
            attr_name (str): The name of the attribute header

        Returns:
            int: Index of the line where the attribute header is found. If it is not found,
                then -1 is returned.
        """
        # search for line that starts with `attr_name`
        # NOTE: due to multiline strings, we need to dedent the lines to make sure we are comparing the correct strings
        # we keep track of the indent bc while it is larger, then we know we are still in the "Attributes" section
        # if the indent is smaller, then we know we are out of the "Attributes" section
        for idx, line in enumerate(docstring_lines):
            if line.startswith(attr_name):
                break
        else:
            return -1

        return idx

    @staticmethod
    def _parse_docstring_attrs(
        docstring: str, attr_name: str, pattern: re.Pattern
    ) -> dict[str, str]:
        lines = docstring.splitlines()

        # search for the attribute header
        idx = DocstringInheritorMetaclass._search_until_attr(lines, attr_name)
        if idx == -1:
            # not found
            return {}

        # get the attributes
        # we need to store both multiline descriptions and the indent of the description
        # the indent is used to determine if we are still in the same attribute, just with
        # a multiline description
        attr_info: dict[str, tuple[list[str], int]] = {}
        curr_attr = ""
        for line in lines[idx + 1 :]:
            line = line.rstrip()
            if not line:
                continue

            dedented = line.lstrip()
            indent = len(line) - len(dedented)

            if indent == 0:  # header ident is 0
                break

            match = pattern.findall(dedented)
            if match:
                match = match[0]
                if len(match) != 2:
                    continue
                # this should basically be the first line of a description
                attr: str
                desc: str
                attr, desc = match
                attr_info[attr] = ([desc], indent)
                curr_attr = attr
            else:
                # if we haven't yet broken check new indent to see if we are still in the same attribute
                attr_indent = attr_info[curr_attr][1]
                if indent > attr_indent:
                    # append to the description
                    attr_info[curr_attr][0].append(dedented)

        attrs = {attr: " ".join(desc) for attr, (desc, _) in attr_info.items()}

        return attrs

    @staticmethod
    def _combine_docstrings(
        clsname: str, docstring: str, attrs: dict[str, str], attr_name: str
    ) -> str:
        if not attrs:
            return docstring

        lines = docstring.splitlines()
        attr_indent_char = " " * DocstringInheritorMetaclass._INDENT
        idx = DocstringInheritorMetaclass._search_until_attr(lines, attr_name)
        if idx == -1:
            if lines:
                # docstring is not empty, so get the preamble
                # add the attributes section
                new_doc_lines = lines[:]
                new_doc_lines.append(f"{attr_name}:")
            else:
                # docstring is empty, so create a default one
                new_doc_lines = f"{clsname} docstring.\n\n{attr_name}:".splitlines()
        else:
            # get the preamble
            new_doc_lines = lines[: idx + 1]

        for attr, desc in attrs.items():
            # TODO: what about multiline descriptions?
            # this is really only for coding visualization, but the docstring renderers handle
            # this already, so I think it's fine to ignore for now
            new_doc_lines.append(f"{attr_indent_char}{attr}: {desc}")
            # NOTE: missing attributes only defined in the based class is handled!

        if idx != -1:
            # get rest of docstring
            skipping_old_attr_section = True
            for line in lines[idx + 1 :]:
                if not line.startswith(" "):
                    # indentation goes back, so we must be at a new section
                    skipping_old_attr_section = False

                if skipping_old_attr_section:
                    continue

                new_doc_lines.append(line)

        return inspect.cleandoc("\n".join(new_doc_lines))


@dataclass
class DataclassValidatorMixin(metaclass=DocstringInheritorMetaclass):
    """Inherit from this for pure `dataclasses.dataclass` validation.

    Subclasses are expected to also be `dataclasses.dataclass`. This also provides
    serialization methods to/from dicts.
    """

    def __post_init__(self):
        validate(self)

    def to_dict(
        self, include: Optional[Set[str]] = None, exclude: Optional[Set[str]] = None
    ):
        return model_dump(self, include, exclude)

    def to_yaml(
        self, include: Optional[Set[str]] = None, exclude: Optional[Set[str]] = None
    ) -> str:
        dict_ser = self.to_dict(include, exclude)

        # now convert to yaml
        buffer = StringIO()

        yaml.dump(dict_ser, buffer)

        return buffer.getvalue()

    @classmethod
    def from_dict(cls, data: KwargType):
        return fromdict(cls, data)

    @classmethod
    def _nested_iter(cls) -> Iterator[Field]:
        stack: list[type] = [cls]
        while stack:
            curr_cls = stack.pop()
            for f in fields(curr_cls):
                if is_dataclass(f.type):
                    stack.append(f.type) # type: ignore
                else:
                    yield f

    @classmethod
    def fields(cls) -> Iterator[str]:
        for f in cls._nested_iter():
            yield f.name

    def clone(self, deep: bool = False):
        fn = copy.deepcopy if deep else copy.copy
        return fn(self)
    
    @classmethod
    def _convert_str_to_enum(cls, init_args: KwargType):
        # convert potential init_args that are str to Enum instances if possible
        for f in cls._nested_iter():
            ftype = f.type
            if isinstance(ftype, type) and issubclass(ftype, Enum):
                init_value = init_args.get(f.name, None)

                # could be a default value so need to check if present
                if init_value is not None:
                    try:
                        enum_type = ftype[init_args[f.name]]
                    except KeyError as e:
                        raise TypeError(
                            f"Invalid enum name for field `{f.name}`: {init_args[f.name]}. Expected one of {list(ftype.__members__)}"
                        ) from e
                    else:
                        init_args[f.name] = enum_type

        return init_args  

def validated_field(default: _T, validator: Callable, *args, **kwargs) -> _T:
    """Create a `dataclasses.field` with a validator.

    This is similar to pydantic's strategy of field validation.
    """
    return field(default=default, *args, metadata={"validator": validator}, **kwargs)

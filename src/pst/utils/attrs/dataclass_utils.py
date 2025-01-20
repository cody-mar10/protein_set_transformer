import copy
import inspect
import re
from enum import Enum
from typing import Iterator, Optional
from warnings import warn

import cattrs
from attrs import Attribute, asdict, define, fields, filters
from attrs import has as is_attrs_dataclass
from cattrs.preconf.pyyaml import PyyamlConverter

from pst.typing import KwargType

_DESERIALIZER_KWARGS: KwargType = {
    # "forbid_extra_keys": True,
    # this is required to opt out of the default cattrs behavior for enums
    "prefer_attrib_converters": True,
}
_DESERIALIZER = cattrs.Converter(**_DESERIALIZER_KWARGS)


def _included_nested_dataclass(model, include: set[str]) -> dict[str, bool]:
    model_fields = fields(type(model))
    is_nested = {}
    for included_field in include:
        field: Attribute = getattr(model_fields, included_field)
        is_nested[included_field] = is_attrs_dataclass(field.type)  # type: ignore[arg-type]

    return is_nested


def model_dump(
    model: object,
    *,
    include: Optional[set[str]] = None,
    exclude: Optional[set[str]] = None,
    convert_enum_to_str: bool = False,
) -> KwargType:
    """Convert an `attrs.define` dataclass to a dictionary. Optionally include or exclude fields
    ONLY from the top-most level. Only `include` OR `exclude` can be used, not both.

    Args:
        model: `attrs.define` dataclass instance
        include (Optional[Set[str]], optional): set of top-level keys to include.
            Defaults to None, meaning include ALL.
        exclude (Optional[Set[str]], optional): set of top-level keys to exclude.
            Defaults to None, meaning exclude NOTHING.
        convert_enum_to_str (bool, optional): convert enum members to their name. Defaults to
            False.

    Raises:
        ValueError: if model is not an `attrs.define` dataclass instance

    Returns:
        KwargType: dictionary representation of the dataclass with all str keys
    """
    if not is_attrs_dataclass(type(model)):
        raise ValueError(f"model must be an `attrs.define` dataclass got {type(model)}")

    if include and exclude:
        raise ValueError("Only include OR exclude can be used, not both")

    asdict_kwargs = {}
    if include is not None:
        asdict_kwargs["filter"] = filters.include(*include)

        # handle special case for dumping nested models
        included_nested_dataclass = _included_nested_dataclass(model, include)
        if any(included_nested_dataclass.values()):
            # need to recurse? or just grab the entire thing?
            # TODO: this only works for one level of nesting
            ser = asdict(model, **asdict_kwargs)
            for key, is_nested in included_nested_dataclass.items():  # pragma: no cover
                if is_nested:
                    value = getattr(model, key)
                    ser[key] = model_dump(value)

            return ser

    elif exclude is not None:
        asdict_kwargs["filter"] = filters.exclude(*exclude)

    if convert_enum_to_str:

        def enum_to_str(instance, field: Attribute, value):
            if isinstance(value, Enum):
                return value.name
            return value

        asdict_kwargs["value_serializer"] = enum_to_str

    return asdict(model, **asdict_kwargs)


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

        curr_attrs = cls._parse_docstring_attrs(curr_doc, attr_name, cls._GOOGLE_DOCSTRING_PATTERN)

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
            clsattrs["__doc__"] = cls._combine_docstrings(name, curr_doc, all_attrs, attr_name)
        return super().__new__(cls, name, bases, clsattrs)

    @staticmethod
    def _get_bases(
        *, cls: Optional[type] = None, bases: Optional[tuple[type, ...]] = None
    ) -> list[type]:  # type: ignore
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
            all_bases.extend(DocstringInheritorMetaclass._get_bases(bases=base.__bases__))

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


@define
class AttrsDataclassUtilitiesMixin(metaclass=DocstringInheritorMetaclass):
    """Inherit from this for pure `attrs.define` dataclass utilities, such as serialization
    and deserialization methods.

    Subclasses are expected to also be `attrs.define`.

    This mixin also features a metaclass that will update the __doc__ attribute of the class
    to include field information from the parent classes. This is useful for inheriting docstrings
    from parent classes AND also enabling users to only document new changes in child classes.
    """

    def to_dict(
        self,
        *,
        include: Optional[set[str]] = None,
        exclude: Optional[set[str]] = None,
        convert_enum_to_str: bool = False,
    ):
        return model_dump(
            self,
            include=include,
            exclude=exclude,
            convert_enum_to_str=convert_enum_to_str,
        )

    def to_yaml(self, **kwargs) -> str:
        if "include" in kwargs or "exclude" in kwargs:
            warn(
                "include and exclude are not officially supported for to_yaml method. "
                "A workaround is to use the `to_dict` method first, and manually convert this "
                "to yaml."
            )
        return PyyamlConverter(**_DESERIALIZER_KWARGS).dumps(self)

    @classmethod
    def from_dict(cls, data: KwargType):
        return _DESERIALIZER.structure(data, cls)

    @classmethod
    def _nested_iter(cls) -> Iterator[Attribute]:
        stack: list[type] = [cls]
        while stack:
            curr_cls = stack.pop()
            f: Attribute
            for f in fields(curr_cls):
                if is_attrs_dataclass(f.type):  # type: ignore
                    stack.append(f.type)
                else:
                    yield f

    @classmethod
    def fields(cls) -> Iterator[str]:
        for f in cls._nested_iter():
            yield f.name

    def clone(self, deep: bool = False):
        fn = copy.deepcopy if deep else copy.copy
        return fn(self)

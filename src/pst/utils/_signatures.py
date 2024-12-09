from typing import Optional, TypeVar, get_type_hints

_T = TypeVar("_T")


def _resolve_config_type_from_init(
    cls, config_name: str = "config", default: Optional[_T] = None
):
    type_hints = get_type_hints(cls.__init__)

    choose_default = (
        # missing annotation, so cant infer type
        config_name not in type_hints
        # generic annotation, ie in base classes, so use default
        # note: shortcircuiting means that this will only be checked
        # if config_name is in type_hints
        or issubclass(type(type_hints[config_name]), TypeVar)
    )

    if choose_default:
        # not annotated or is generic
        if default is None:
            raise ValueError(
                f"Expected a default type annotation for 'config' in {cls.__name__} since the"
                f"type of the config variable {config_name} cannot be inferred or is a generic"
            )

        return default

    # TODO: what if it is a Union
    # actually I don't think that matters. The caller should handle that

    return type_hints[config_name]

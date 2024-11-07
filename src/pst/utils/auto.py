from typing import Literal, Type, overload

from pst.nn.base import _BaseProteinSetTransformer
from pst.nn.modules import (
    MLMProteinSetTransformer,
    ProteinSetTransformer,
    ProteinSetTransformerEncoder,
)
from pst.utils.cli.experiment import ModelType as _ModelType


@overload
def auto_resolve_model_type(
    model_type: Literal["ProteinSetTransformer"],
) -> Type[ProteinSetTransformer]: ...


@overload
def auto_resolve_model_type(
    model_type: Literal["ProteinSetTransformerEncoder"],
) -> Type[ProteinSetTransformerEncoder]: ...


@overload
def auto_resolve_model_type(
    model_type: Literal["MLMProteinSetTransformer"],
) -> Type[MLMProteinSetTransformer]: ...


def auto_resolve_model_type(model_type: _ModelType) -> Type[_BaseProteinSetTransformer]:
    if model_type == "ProteinSetTransformer":
        return ProteinSetTransformer

    if model_type == "ProteinSetTransformerEncoder":
        return ProteinSetTransformerEncoder

    if model_type == "MLMProteinSetTransformer":
        return MLMProteinSetTransformer

    raise ValueError(f"Unknown model type: {model_type}")

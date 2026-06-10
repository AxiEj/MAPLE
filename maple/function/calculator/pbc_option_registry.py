"""Shared model-option validation registry for official PBC-capable backends."""

from typing import Callable, Mapping, Optional

from .aimnet.options import AIMNET_LEGACY_MODELS, AIMNET_PBC_MODELS, validate_aimnet_options
from .mace.options import (
    MACE_PBC_MODELS,
    MACEPOL_PBC_MODELS,
    validate_mace_pbc_options,
    validate_macepol_pbc_options,
)

ModelOptionValidator = Callable[[Optional[Mapping[str, object]]], dict[str, object]]


def _validate_aimnet_legacy(model_options: Optional[Mapping[str, object]]) -> dict[str, object]:
    return validate_aimnet_options(model_options, pbc=False)


def _validate_aimnet_pbc(model_options: Optional[Mapping[str, object]]) -> dict[str, object]:
    return validate_aimnet_options(model_options, pbc=True)


PBC_OPTION_VALIDATORS: tuple[tuple[frozenset[str], ModelOptionValidator], ...] = (
    (AIMNET_LEGACY_MODELS, _validate_aimnet_legacy),
    (frozenset(AIMNET_PBC_MODELS), _validate_aimnet_pbc),
    (frozenset(MACE_PBC_MODELS), validate_mace_pbc_options),
    (frozenset(MACEPOL_PBC_MODELS), validate_macepol_pbc_options),
)


def validate_model_pbc_options(
    model: str,
    model_options: Optional[Mapping[str, object]],
) -> dict[str, object]:
    for model_names, validator in PBC_OPTION_VALIDATORS:
        if model in model_names:
            return validator(model_options)
    return {}

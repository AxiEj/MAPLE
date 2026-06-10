"""MACE official PBC backend option contracts."""

from typing import Mapping, Optional


MACE_PBC_MODELS = {
    "mace-mp-pbc-small": "mace-mp-small",
    "mace-mp-pbc-medium": "mace-mp-medium",
    "mace-mp-pbc-large": "mace-mp-large",
}
MACE_PBC_OFFICIAL_FOUNDATIONS = {
    "mace-mp-small": "small",
    "mace-mp-medium": "medium",
    "mace-mp-large": "large",
}
MACE_PBC_FOUNDATIONS = frozenset(MACE_PBC_OFFICIAL_FOUNDATIONS)
MACE_PBC_OPTION_KEYS = frozenset(
    {"hessian", "foundation", "default_dtype", "dispersion", "head"}
)
MACE_PBC_DTYPES = frozenset({"float32", "float64"})

MACEPOL_PBC_MODELS = {
    "macepol-pbc-small": "polar-1-s",
    "macepol-pbc-medium": "polar-1-m",
    "macepol-pbc-large": "polar-1-l",
}
MACEPOL_PBC_OPTION_KEYS = frozenset({"hessian", "default_dtype"})
MACEPOL_PBC_DTYPES = frozenset({"float32", "float64"})


def validate_mace_pbc_options(model_options: Optional[Mapping[str, object]]) -> dict[str, object]:
    """Validate official MACE PBC options and return normalized values."""

    if not model_options:
        return {}

    unknown = sorted(set(model_options) - MACE_PBC_OPTION_KEYS)
    if unknown:
        unknown_text = ", ".join(unknown)
        supported_text = ", ".join(sorted(MACE_PBC_OPTION_KEYS))
        raise ValueError(
            f"Unsupported MACE PBC option(s): {unknown_text}. "
            f"Supported options: {supported_text}"
        )

    options: dict[str, object] = {}
    if "hessian" in model_options:
        options["hessian"] = model_options["hessian"]

    foundation = model_options.get("foundation")
    if foundation is not None:
        foundation = str(foundation).lower()
        if foundation not in MACE_PBC_FOUNDATIONS:
            supported_text = ", ".join(sorted(MACE_PBC_FOUNDATIONS))
            raise ValueError(
                f"Unsupported MACE PBC foundation: '{foundation}'. "
                f"Supported foundations: {supported_text}"
            )
        options["foundation"] = foundation

    default_dtype = str(model_options.get("default_dtype", "float32")).lower()
    if default_dtype not in MACE_PBC_DTYPES:
        supported_text = ", ".join(sorted(MACE_PBC_DTYPES))
        raise ValueError(
            f"Unsupported MACE PBC default_dtype: '{default_dtype}'. "
            f"Supported values: {supported_text}"
        )
    options["default_dtype"] = default_dtype

    dispersion = model_options.get("dispersion", False)
    if not isinstance(dispersion, bool):
        raise ValueError("MACE PBC option 'dispersion' must be true or false.")
    options["dispersion"] = dispersion

    head = model_options.get("head")
    if head is not None:
        head = str(head)
        if not head:
            raise ValueError("MACE PBC option 'head' must be non-empty.")
        options["head"] = head

    return options


def validate_macepol_pbc_options(model_options: Optional[Mapping[str, object]]) -> dict[str, object]:
    """Validate official MACE-Polar PBC options and return normalized values."""

    if not model_options:
        return {}

    unknown = sorted(set(model_options) - MACEPOL_PBC_OPTION_KEYS)
    if unknown:
        unknown_text = ", ".join(unknown)
        supported_text = ", ".join(sorted(MACEPOL_PBC_OPTION_KEYS))
        raise ValueError(
            f"Unsupported MACE-Polar PBC option(s): {unknown_text}. "
            f"Supported options: {supported_text}"
        )

    options: dict[str, object] = {}
    if "hessian" in model_options:
        options["hessian"] = model_options["hessian"]

    default_dtype = str(model_options.get("default_dtype", "float32")).lower()
    if default_dtype not in MACEPOL_PBC_DTYPES:
        supported_text = ", ".join(sorted(MACEPOL_PBC_DTYPES))
        raise ValueError(
            f"Unsupported MACE-Polar PBC default_dtype: '{default_dtype}'. "
            f"Supported values: {supported_text}"
        )
    options["default_dtype"] = default_dtype

    return options

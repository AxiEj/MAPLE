"""AIMNet calculator option contracts shared by parser and factory code."""

from typing import Mapping, Optional


AIMNET_LEGACY_MODELS = frozenset({"aimnet2", "aimnet2nse"})
AIMNET_COULOMB_OPTION_KEYS = frozenset(
    {"coulomb", "coulomb_method", "cutoff", "dsf_alpha"}
)
AIMNET_LEGACY_OPTION_KEYS = AIMNET_COULOMB_OPTION_KEYS | frozenset({"hessian"})
AIMNET_LEGACY_COULOMB_METHODS = frozenset({"simple", "dsf"})
AIMNET_PBC_MODELS = {
    "aimnet2-pbc": "aimnet2",
    "aimnet2nse-pbc": "aimnet2-nse",
}
AIMNET_PBC_OPTION_KEYS = AIMNET_COULOMB_OPTION_KEYS | frozenset(
    {"ewald_accuracy", "pme_cutoff"}
)
AIMNET_PBC_COULOMB_METHODS = frozenset({"dsf", "ewald", "pme"})


def validate_aimnet_options(
    model_options: Optional[Mapping[str, object]],
    *,
    pbc: bool = False,
) -> dict[str, object]:
    """Validate AIMNet model options and return normalized Coulomb settings."""

    if not model_options:
        return {}

    allowed_keys = AIMNET_PBC_OPTION_KEYS if pbc else AIMNET_LEGACY_OPTION_KEYS
    allowed_coulomb = AIMNET_PBC_COULOMB_METHODS if pbc else AIMNET_LEGACY_COULOMB_METHODS
    label = "AIMNet2 PBC" if pbc else "AIMNet2"

    unknown = sorted(set(model_options) - allowed_keys)
    if unknown:
        unknown_text = ", ".join(unknown)
        supported_text = ", ".join(sorted(allowed_keys))
        raise ValueError(
            f"Unsupported {label} option(s): {unknown_text}. "
            f"Supported options: {supported_text}"
        )

    options: dict[str, object] = {}
    if "hessian" in model_options:
        options["hessian"] = model_options["hessian"]

    coulomb = model_options.get("coulomb")
    coulomb_method = model_options.get("coulomb_method")
    if coulomb is not None or coulomb_method is not None:
        normalized = {
            key: str(value).lower()
            for key, value in (
                ("coulomb", coulomb),
                ("coulomb_method", coulomb_method),
            )
            if value is not None
        }
        if len(set(normalized.values())) > 1:
            raise ValueError(
                f"Conflicting {label} Coulomb options: "
                f"coulomb={coulomb!r}, coulomb_method={coulomb_method!r}. "
                "Specify only one spelling or use matching values."
            )
        method = next(iter(normalized.values()))
        if method not in allowed_coulomb:
            supported_text = ", ".join(sorted(allowed_coulomb))
            raise ValueError(
                f"Unsupported {label} Coulomb method: '{method}'. "
                f"Supported methods: {supported_text}"
            )
        options["coulomb"] = method

    for key in ("cutoff", "dsf_alpha", "ewald_accuracy", "pme_cutoff"):
        if key in model_options:
            value = float(model_options[key])
            if value <= 0:
                raise ValueError(f"{label} option '{key}' must be positive.")
            options[key] = value

    return options

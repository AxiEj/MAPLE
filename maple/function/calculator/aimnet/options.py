"""AIMNet calculator option contracts shared by parser and factory code."""

from typing import Mapping, Optional


AIMNET_LEGACY_MODELS = frozenset({"aimnet2", "aimnet2nse"})
AIMNET_LEGACY_OPTION_KEYS = frozenset({"hessian", "coulomb", "cutoff", "dsf_alpha"})
AIMNET_LEGACY_COULOMB_METHODS = frozenset({"simple", "dsf", "ewald"})
AIMNET_PBC_MODELS = {
    "aimnet2-pbc": "aimnet2",
    "aimnet2nse-pbc": "aimnet2-nse",
}
AIMNET_PBC_OPTION_KEYS = AIMNET_LEGACY_OPTION_KEYS | frozenset(
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
    if coulomb is not None:
        coulomb = str(coulomb).lower()
        if coulomb not in allowed_coulomb:
            supported_text = ", ".join(sorted(allowed_coulomb))
            raise ValueError(
                f"Unsupported {label} Coulomb method: '{coulomb}'. "
                f"Supported methods: {supported_text}"
            )
        options["coulomb"] = coulomb

    for key in ("cutoff", "dsf_alpha", "ewald_accuracy", "pme_cutoff"):
        if key in model_options:
            value = float(model_options[key])
            if value <= 0:
                raise ValueError(f"{label} option '{key}' must be positive.")
            options[key] = value

    return options

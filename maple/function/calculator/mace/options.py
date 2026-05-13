"""MACE official PBC backend option contracts."""

MACE_PBC_MODELS = {
    "mace-mp-pbc": "medium-mpa-0",
    "mace-omat-pbc": "medium-omat-0",
    "mace-matpes-pbc": "mace-matpes-r2scan-0",
    "mace-mh-pbc": "mh-1",
}

MACE_PBC_DEFAULT_HEADS = {
    "mace-mh-pbc": "omat_pbe",
}

MACE_PBC_OPTION_KEYS = {"hessian", "foundation", "default_dtype", "dispersion", "head"}
MACE_PBC_DTYPES = {"float32", "float64"}

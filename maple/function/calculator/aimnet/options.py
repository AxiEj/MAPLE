"""AIMNet calculator option contracts shared by parser and factory code."""

AIMNET_LEGACY_MODELS = {"aimnet2", "aimnet2nse"}
AIMNET_LEGACY_OPTION_KEYS = {"hessian", "coulomb", "cutoff", "dsf_alpha"}
AIMNET_LEGACY_COULOMB_METHODS = {"simple", "dsf", "ewald"}

AIMNET_PBC_MODELS = {
    "aimnet2-pbc": "aimnet2",
    "aimnet2nse-pbc": "aimnet2-nse",
}
AIMNET_PBC_OPTION_KEYS = {
    "hessian",
    "coulomb",
    "cutoff",
    "dsf_alpha",
    "ewald_accuracy",
}
AIMNET_PBC_COULOMB_METHODS = {"dsf", "ewald", "pme"}

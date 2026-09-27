"""Framework parameter ownership declarations without runtime dependencies."""


GLOBAL_PARAMETER_KEYS = frozenset({
    "model",
    "model_options",
    "device",
    "gpuid",
    "d4",
    "pbc",
    "solv",
    "level",
})

# ``mdp`` records how the MD frontend resolved its input. It is accepted at
# the framework-to-MD boundary, but is not a global keyword for other jobs.
MD_FRONTEND_PROVENANCE_KEYS = frozenset({"mdp"})


__all__ = ["GLOBAL_PARAMETER_KEYS", "MD_FRONTEND_PROVENANCE_KEYS"]

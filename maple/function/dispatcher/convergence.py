from types import MappingProxyType


# Gaussian-style force thresholds use Eh/Å; displacement thresholds use Å.
CONVERGENCE_THRESHOLDS = MappingProxyType({
    "extratight": MappingProxyType({
        "f_max_th": 0.00030,
        "f_rms_th": 0.00020,
        "dp_max_th": 0.00030,
        "dp_rms_th": 0.00020,
    }),
    "tight": MappingProxyType({
        "f_max_th": 0.00085,
        "f_rms_th": 0.00055,
        "dp_max_th": 0.00110,
        "dp_rms_th": 0.00075,
    }),
    "medium": MappingProxyType({
        "f_max_th": 0.00285,
        "f_rms_th": 0.00190,
        "dp_max_th": 0.00315,
        "dp_rms_th": 0.00210,
    }),
    "loose": MappingProxyType({
        "f_max_th": 0.00380,
        "f_rms_th": 0.00250,
        "dp_max_th": 0.00600,
        "dp_rms_th": 0.00400,
    }),
    "extraloose": MappingProxyType({
        "f_max_th": 0.00755,
        "f_rms_th": 0.00500,
        "dp_max_th": 0.01200,
        "dp_rms_th": 0.00800,
    }),
    "superloose": MappingProxyType({
        "f_max_th": 0.08500,
        "f_rms_th": 0.05500,
        "dp_max_th": 0.14500,
        "dp_rms_th": 0.09500,
    }),
})

DEFAULT_CONVERGENCE_LEVEL = "medium"
CONVERGENCE_CONTEXT_KEYS = frozenset(
    CONVERGENCE_THRESHOLDS[DEFAULT_CONVERGENCE_LEVEL]
)


def convergence_thresholds(level):
    if isinstance(level, str):
        return CONVERGENCE_THRESHOLDS.get(
            level, CONVERGENCE_THRESHOLDS[DEFAULT_CONVERGENCE_LEVEL]
        )
    return CONVERGENCE_THRESHOLDS[DEFAULT_CONVERGENCE_LEVEL]

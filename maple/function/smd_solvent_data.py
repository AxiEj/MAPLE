"""Vendored PySCF 2.13.1 SMD solvent descriptor snapshot.

The descriptor table is copied from ``pyscf.solvent.smd.solvent_db`` in
PySCF 2.13.1 so runtime solvent lookup does not import PySCF.  PySCF is
distributed under the Apache License 2.0; see
https://github.com/pyscf/pyscf/blob/v2.13.1/LICENSE.

Source module SHA-256: 287d43f251a16c50791f35fcf43eba12b92bd2b5aa0b3c739a8e3eb465c0ccfc
Normalized table SHA-256: 0353111ce924f14f955e4529ea1adb95c705b98159e8b850a7131d601483c6d0

Each record is ``(canonical_name, pyscf_name, eight_descriptors)``.  The empty
all-zero sentinel in upstream ``solvent_db`` is intentionally excluded.
"""

from __future__ import annotations

PYSCF_SMD_SOURCE_VERSION = "2.13.1"
PYSCF_SMD_SOURCE_MODULE_SHA256 = (
    "287d43f251a16c50791f35fcf43eba12b92bd2b5aa0b3c739a8e3eb465c0ccfc"
)
PYSCF_SMD_TABLE_SHA256 = (
    "0353111ce924f14f955e4529ea1adb95c705b98159e8b850a7131d601483c6d0"
)

SMD_SOLVENT_DESCRIPTOR_SNAPSHOT: tuple[
    tuple[str, str, tuple[float, float, float, float, float, float, float, float]],
    ...,
] = (
    (
        "1-1-1-trichloroethane",
        "1,1,1-trichloroethane",
        (1.4379, 1.4313, 0.0, 0.09, 36.24, 7.0826, 0.0, 0.6),
    ),
    (
        "1-1-2-trichloroethane",
        "1,1,2-trichloroethane",
        (1.4717, 1.4689, 0.13, 0.13, 48.97, 7.1937, 0.0, 0.6),
    ),
    (
        "1-2-4-trimethylbenzene",
        "1,2,4-trimethylbenzene",
        (1.5048, 1.5024, 0.0, 0.19, 42.03, 2.3653, 0.667, 0.0),
    ),
    (
        "1-2-dibromoethane",
        "1,2-dibromoethane",
        (1.5387, 1.5364, 0.1, 0.17, 56.93, 4.9313, 0.0, 0.5),
    ),
    (
        "1-2-dichloroethane",
        "1,2-dichloroethane",
        (1.4448, 1.4425, 0.1, 0.11, 45.86, 10.125, 0.0, 0.5),
    ),
    (
        "1-2-ethanediol",
        "1,2-ethanediol",
        (1.4318, 1.4306, 0.58, 0.78, 69.07, 40.245, 0.0, 0.5),
    ),
    (
        "1-4-dioxane",
        "1,4-dioxane",
        (1.4224, 1.4204, 0.0, 0.64, 47.14, 2.2099, 0.0, 0.0),
    ),
    (
        "1-bromo-2-methylpropane",
        "1-bromo-2-methylpropane",
        (1.4348, 1.4349, 0.0, 0.12, 34.69, 7.7792, 0.0, 0.2),
    ),
    (
        "1-bromooctane",
        "1-bromooctane",
        (1.4524, 1.45, 0.0, 0.12, 41.28, 5.0244, 0.0, 0.111),
    ),
    (
        "1-bromopentane",
        "1-bromopentane",
        (1.4447, 1.442, 0.0, 0.12, 38.7, 6.269, 0.0, 0.167),
    ),
    (
        "1-bromopropane",
        "1-bromopropane",
        (1.4343, 1.4315, 0.0, 0.12, 36.36, 8.0496, 0.0, 0.25),
    ),
    ("1-butanol", "1-butanol", (1.3993, 1.3971, 0.37, 0.48, 35.88, 17.332, 0.0, 0.0)),
    (
        "1-chlorohexane",
        "1-chlorohexane",
        (1.4199, -1, 0.0, 0.1, 37.03, 5.9491, 0.0, 0.143),
    ),
    (
        "1-chloropentane",
        "1-chloropentane",
        (1.4127, 1.4104, 0.0, 0.1, 35.12, 6.5022, 0.0, 0.167),
    ),
    (
        "1-chloropropane",
        "1-chloropropane",
        (1.3879, 1.3851, 0.0, 0.1, 30.66, 8.3548, 0.0, 0.25),
    ),
    ("1-decanol", "1-decanol", (1.4372, 1.4353, 0.37, 0.48, 41.04, 7.5305, 0.0, 0.0)),
    (
        "1-fluorooctane",
        "1-fluorooctane",
        (1.3935, 1.3927, 0.0, 0.1, 33.92, 3.89, 0.0, 0.111),
    ),
    ("1-heptanol", "1-heptanol", (1.4249, 1.4224, 0.37, 0.48, 38.5, 11.321, 0.0, 0.0)),
    ("1-hexanol", "1-hexanol", (1.4178, 1.4162, 0.37, 0.48, 37.15, 12.51, 0.0, 0.0)),
    ("1-hexene", "1-hexene", (1.3837, 1.385, 0.0, 0.07, 25.76, 2.0717, 0.0, 0.0)),
    ("1-hexyne", "1-hexyne", (1.3989, 1.3957, 0.12, 0.1, 28.79, 2.615, 0.0, 0.0)),
    (
        "1-iodobutane",
        "1-iodobutane",
        (1.5001, 1.4958, 0.0, 0.15, 40.65, 6.173, 0.0, 0.0),
    ),
    (
        "1-iodohexadecane",
        "1-iodohexadecane",
        (1.4806, -1, 0.0, 0.15, 46.48, 3.5338, 0.0, 0.0),
    ),
    (
        "1-iodopentane",
        "1-iodopentane",
        (1.4959, -1, 0.0, 0.15, 41.56, 5.6973, 0.0, 0.0),
    ),
    (
        "1-iodopropane",
        "1-iodopropane",
        (1.5058, 1.5027, 0.0, 0.15, 41.45, 6.9626, 0.0, 0.0),
    ),
    (
        "1-nitropropane",
        "1-nitropropane",
        (1.4018, 1.3996, 0.0, 0.31, 43.32, 23.73, 0.0, 0.0),
    ),
    ("1-nonanol", "1-nonanol", (1.4333, 1.4319, 0.37, 0.48, 40.14, 8.5991, 0.0, 0.0)),
    ("1-octanol", "1-octanol", (1.4295, 1.4279, 0.37, 0.48, 39.01, 9.8629, 0.0, 0.0)),
    ("1-pentanol", "1-pentanol", (1.4101, 1.408, 0.37, 0.48, 36.5, 15.13, 0.0, 0.0)),
    ("1-pentene", "1-pentene", (1.3715, 1.3684, 0.0, 0.07, 22.24, 1.9905, 0.0, 0.0)),
    ("1-propanol", "1-propanol", (1.385, 1.3837, 0.37, 0.48, 33.57, 20.524, 0.0, 0.0)),
    (
        "2-2-2-trifluoroethanol",
        "2,2,2-trifluoroethanol",
        (1.2907, -1, 0.57, 0.25, 42.02, 26.726, 0.0, 0.5),
    ),
    (
        "2-2-4-trimethylpentane",
        "2,2,4-trimethylpentane",
        (1.3915, 1.3889, 0.0, 0.0, 26.38, 1.9358, 0.0, 0.0),
    ),
    (
        "2-4-dimethylpentane",
        "2,4-dimethylpentane",
        (1.3815, 1.3788, 0.0, 0.0, 25.42, 1.8939, 0.0, 0.0),
    ),
    (
        "2-4-dimethylpyridine",
        "2,4-dimethylpyridine",
        (1.501, 1.4985, 0.0, 0.63, 46.86, 9.4176, 0.625, 0.0),
    ),
    (
        "2-6-dimethylpyridine",
        "2,6-dimethylpyridine",
        (1.4953, 1.4952, 0.0, 0.63, 44.64, 7.1735, 0.625, 0.0),
    ),
    (
        "2-bromopropane",
        "2-bromopropane",
        (1.4251, 1.4219, 0.0, 0.14, 33.46, 9.361, 0.0, 0.25),
    ),
    ("2-butanol", "2-butanol", (1.3978, 1.3949, 0.33, 0.56, 32.44, 15.944, 0.0, 0.0)),
    (
        "2-chlorobutane",
        "2-chlorobutane",
        (1.3971, 1.3941, 0.0, 0.12, 31.1, 8.393, 0.0, 0.2),
    ),
    ("2-heptanone", "2-heptanone", (1.4088, 1.4073, 0.0, 0.51, 37.6, 11.658, 0.0, 0.0)),
    ("2-hexanone", "2-hexanone", (1.4007, 1.3987, 0.0, 0.51, 36.63, 14.136, 0.0, 0.0)),
    (
        "2-methoxyethanol",
        "2-methoxyethanol",
        (1.4024, 1.4003, 0.3, 0.84, 44.39, 17.2, 0.0, 0.0),
    ),
    (
        "2-methyl-1-propanol",
        "2-methyl-1-propanol",
        (1.3955, 1.3938, 0.37, 0.48, 32.38, 16.777, 0.0, 0.0),
    ),
    (
        "2-methyl-2-propanol",
        "2-methyl-2-propanol",
        (1.3878, 1.3852, 0.31, 0.6, 28.73, 12.47, 0.0, 0.0),
    ),
    (
        "2-methylpentane",
        "2-methylpentane",
        (1.3715, 1.3687, 0.0, 0.0, 24.3, 1.89, 0.0, 0.0),
    ),
    (
        "2-methylpyridine",
        "2-methylpyridine",
        (1.4957, 1.4984, 0.0, 0.58, 47.5, 9.9533, 0.714, 0.0),
    ),
    (
        "2-nitropropane",
        "2-nitropropane",
        (1.3944, 1.3923, 0.0, 0.33, 42.16, 25.654, 0.0, 0.0),
    ),
    ("2-octanone", "2-octanone", (1.4151, 1.4133, 0.0, 0.51, 37.29, 9.4678, 0.0, 0.0)),
    ("2-pentanone", "2-pentanone", (1.3895, 1.3885, 0.0, 0.51, 33.46, 15.2, 0.0, 0.0)),
    ("2-propanol", "2-propanol", (1.3776, 1.3752, 0.33, 0.56, 30.13, 19.264, 0.0, 0.0)),
    (
        "2-propen-1-ol",
        "2-propen-1-ol",
        (1.4135, 0.0, 0.38, 0.48, 36.39, 19.011, 0.0, 0.0),
    ),
    ("e-2-pentene", "E-2-pentene", (1.3793, 1.3761, 0.0, 0.07, 23.62, 2.051, 0.0, 0.0)),
    (
        "3-methylpyridine",
        "3-methylpyridine",
        (1.504, 1.5043, 0.0, 0.54, 49.61, 11.645, 0.714, 0.0),
    ),
    ("3-pentanone", "3-pentanone", (1.3924, 1.3905, 0.0, 0.51, 35.61, 16.78, 0.0, 0.0)),
    (
        "4-heptanone",
        "4-heptanone",
        (1.4069, 1.4045, 0.0, 0.51, 35.98, 12.257, 0.0, 0.0),
    ),
    (
        "4-methyl-2-pentanone",
        "4-methyl-2-pentanone",
        (1.3962, 1.394, 0.0, 0.51, 33.83, 12.887, 0.0, 0.0),
    ),
    (
        "4-methylpyridine",
        "4-methylpyridine",
        (1.5037, 1.503, 0.0, 0.54, 50.17, 11.957, 0.714, 0.0),
    ),
    ("5-nonanone", "5-nonanone", (1.4195, 0.0, 0.0, 0.51, 37.83, 10.6, 0.0, 0.0)),
    (
        "acetic-acid",
        "acetic acid",
        (1.372, 1.3698, 0.61, 0.44, 39.01, 6.2528, 0.0, 0.0),
    ),
    ("acetone", "acetone", (1.3588, 1.3559, 0.04, 0.49, 33.77, 20.493, 0.0, 0.0)),
    (
        "acetonitrile",
        "acetonitrile",
        (1.3442, 1.3416, 0.07, 0.32, 41.25, 35.688, 0.0, 0.0),
    ),
    (
        "acetophenone",
        "acetophenone",
        (1.5372, 1.5321, 0.0, 0.48, 56.19, 17.44, 0.667, 0.0),
    ),
    ("aniline", "aniline", (1.5863, 1.5834, 0.26, 0.41, 60.62, 6.8882, 0.857, 0.0)),
    ("anisole", "anisole", (1.5174, 1.5143, 0.0, 0.29, 50.52, 4.2247, 0.75, 0.0)),
    (
        "benzaldehyde",
        "benzaldehyde",
        (1.5463, 1.5433, 0.0, 0.39, 54.69, 18.22, 0.857, 0.0),
    ),
    ("benzene", "benzene", (1.5011, 1.4972, 0.0, 0.14, 40.62, 2.2706, 1.0, 0.0)),
    (
        "benzonitrile",
        "benzonitrile",
        (1.5289, 1.5257, 0.0, 0.33, 55.83, 25.592, 0.75, 0.0),
    ),
    (
        "benzylalcohol",
        "benzylalcohol",
        (1.5396, 1.5384, 0.33, 0.56, 52.96, 12.457, 0.75, 0.0),
    ),
    (
        "bromobenzene",
        "bromobenzene",
        (1.5597, 1.5576, 0.0, 0.09, 50.72, 5.3954, 0.857, 0.143),
    ),
    ("bromoethane", "bromoethane", (1.4239, 1.4187, 0.0, 0.12, 34.0, 9.01, 0.0, 0.333)),
    ("bromoform", "bromoform", (1.6005, 1.5956, 0.15, 0.06, 64.58, 4.2488, 0.0, 0.75)),
    ("butanal", "butanal", (1.3843, 1.3766, 0.0, 0.45, 35.06, 13.45, 0.0, 0.0)),
    (
        "butanoic-acid",
        "butanoic acid",
        (1.398, 1.3958, 0.6, 0.45, 37.49, 2.9931, 0.0, 0.0),
    ),
    ("butanone", "butanone", (1.3788, 1.3764, 0.0, 0.51, 34.5, 18.246, 0.0, 0.0)),
    (
        "butanonitrile",
        "butanonitrile",
        (1.3842, 1.382, 0.0, 0.36, 38.75, 24.291, 0.0, 0.0),
    ),
    (
        "butylethanoate",
        "butylethanoate",
        (1.3941, 1.3923, 0.0, 0.45, 35.81, 4.9941, 0.0, 0.0),
    ),
    ("butylamine", "butylamine", (1.4031, 1.3987, 0.16, 0.61, 33.74, 4.6178, 0.0, 0.0)),
    (
        "n-butylbenzene",
        "n-butylbenzene",
        (1.4898, 1.4874, 0.0, 0.15, 41.33, 2.36, 0.6, 0.0),
    ),
    (
        "sec-butylbenzene",
        "sec-butylbenzene",
        (1.4895, 1.4878, 0.0, 0.16, 40.35, 2.3446, 0.6, 0.0),
    ),
    (
        "tert-butylbenzene",
        "tert-butylbenzene",
        (1.4927, 1.4902, 0.0, 0.16, 39.78, 2.3447, 0.6, 0.0),
    ),
    (
        "carbon-disulfide",
        "carbon disulfide",
        (1.6319, 1.6241, 0.0, 0.07, 45.45, 2.6105, 0.0, 0.0),
    ),
    (
        "carbon-tetrachloride",
        "carbon tetrachloride",
        (1.4601, 1.4574, 0.0, 0.0, 38.04, 2.228, 0.0, 0.8),
    ),
    (
        "chlorobenzene",
        "chlorobenzene",
        (1.5241, 1.5221, 0.0, 0.07, 47.48, 5.6968, 0.857, 0.143),
    ),
    (
        "chloroform",
        "chloroform",
        (1.4459, 1.4431, 0.15, 0.02, 38.39, 4.7113, 0.0, 0.75),
    ),
    (
        "a-chlorotoluene",
        "a-chlorotoluene",
        (1.5391, 0.0, 0.0, 0.33, 53.04, 6.7175, 0.75, 0.125),
    ),
    (
        "o-chlorotoluene",
        "o-chlorotoluene",
        (1.5268, 1.5233, 0.0, 0.07, 47.43, 4.6331, 0.75, 0.125),
    ),
    ("m-cresol", "m-cresol", (1.5438, 1.5394, 0.57, 0.34, 51.37, 12.44, 0.75, 0.0)),
    ("o-cresol", "o-cresol", (1.5361, 1.5399, 0.52, 0.3, 53.11, 6.76, 0.75, 0.0)),
    ("cyclohexane", "cyclohexane", (1.4266, 1.4235, 0.0, 0.0, 35.48, 2.0165, 0.0, 0.0)),
    (
        "cyclohexanone",
        "cyclohexanone",
        (1.4507, 1.4507, 0.0, 0.56, 49.76, 15.619, 0.0, 0.0),
    ),
    (
        "cyclopentane",
        "cyclopentane",
        (1.4065, 1.4036, 0.0, 0.0, 31.49, 1.9608, 0.0, 0.0),
    ),
    ("cyclopentanol", "cyclopentanol", (1.453, -1, 0.32, 0.56, 46.8, 16.989, 0.0, 0.0)),
    (
        "cyclopentanone",
        "cyclopentanone",
        (1.4366, 1.4347, 0.0, 0.52, 47.21, 13.58, 0.0, 0.0),
    ),
    (
        "decalin-cis-trans-mixture",
        "decalin (cis/trans mixture)",
        (1.4753, 1.472, 0.0, 0.0, 43.82, 2.196, 0.0, 0.0),
    ),
    ("cis-decalin", "cis-decalin", (1.481, 1.4788, 0.0, 0.0, 45.45, 2.2139, 0.0, 0.0)),
    ("n-decane", "n-decane", (1.4102, 1.4094, 0.0, 0.0, 33.64, 1.9846, 0.0, 0.0)),
    (
        "dibromomethane",
        "dibromomethane",
        (1.542, 1.5389, 0.1, 0.1, 56.21, 7.2273, 0.0, 0.667),
    ),
    ("butylether", "butylether", (1.3992, 1.3968, 0.0, 0.45, 35.98, 3.0473, 0.0, 0.0)),
    (
        "o-dichlorobenzene",
        "o-dichlorobenzene",
        (1.5515, 1.5491, 0.0, 0.04, 52.72, 9.9949, 0.75, 0.25),
    ),
    (
        "e-1-2-dichloroethene",
        "E-1,2-dichloroethene",
        (1.4454, 1.4435, 0.09, 0.05, 37.13, 2.14, 0.0, 0.5),
    ),
    (
        "z-1-2-dichloroethene",
        "Z-1,2-dichloroethene",
        (1.449, 1.4461, 0.11, 0.05, 39.8, 9.2, 0.0, 0.5),
    ),
    (
        "dichloromethane",
        "dichloromethane",
        (1.4242, 1.4212, 0.1, 0.05, 39.15, 8.93, 0.0, 0.667),
    ),
    (
        "diethylether",
        "diethylether",
        (1.3526, 1.3496, 0.0, 0.41, 23.96, 4.24, 0.0, 0.0),
    ),
    (
        "diethylsulfide",
        "diethylsulfide",
        (1.443, 1.4401, 0.0, 0.32, 35.36, 5.723, 0.0, 0.0),
    ),
    (
        "diethylamine",
        "diethylamine",
        (1.3864, 1.3825, 0.08, 0.69, 28.57, 3.5766, 0.0, 0.0),
    ),
    (
        "diiodomethane",
        "diiodomethane",
        (1.7425, 1.738, 0.05, 0.23, 95.25, 5.32, 0.0, 0.0),
    ),
    (
        "diisopropyl-ether",
        "diisopropyl ether",
        (1.3679, 1.3653, 0.0, 0.41, 24.86, 3.38, 0.0, 0.0),
    ),
    (
        "cis-1-2-dimethylcyclohexane",
        "cis-1,2-dimethylcyclohexane",
        (1.436, 1.4336, 0.0, 0.0, 36.28, 2.06, 0.0, 0.0),
    ),
    (
        "dimethyldisulfide",
        "dimethyldisulfide",
        (1.5289, 1.522, 0.0, 0.28, 48.06, 9.6, 0.0, 0.0),
    ),
    (
        "n-n-dimethylacetamide",
        "N,N-dimethylacetamide",
        (1.438, 1.4358, 0.0, 0.78, 47.62, 37.781, 0.0, 0.0),
    ),
    (
        "dimethylformamide",
        "N,N-dimethylformamide",
        (1.4305, 1.428, 0.0, 0.74, 49.56, 37.219, 0.0, 0.0),
    ),
    (
        "dimethylsulfoxide",
        "dimethylsulfoxide",
        (1.4783, 1.4783, 0.0, 0.88, 61.78, 46.826, 0.0, 0.0),
    ),
    ("diphenylether", "diphenylether", (1.5787, -1, 0.0, 0.2, 38.5, 3.73, 0.923, 0.0)),
    (
        "dipropylamine",
        "dipropylamine",
        (1.405, 1.4018, 0.08, 0.69, 32.11, 2.9112, 0.0, 0.0),
    ),
    ("n-dodecane", "n-dodecane", (1.4216, 1.4151, 0.0, 0.0, 35.85, 2.006, 0.0, 0.0)),
    ("ethanethiol", "ethanethiol", (1.431, 1.4278, 0.0, 0.24, 33.22, 6.667, 0.0, 0.0)),
    ("ethanol", "ethanol", (1.3611, 1.3593, 0.37, 0.48, 31.62, 24.852, 0.0, 0.0)),
    (
        "ethylethanoate",
        "ethylethanoate",
        (1.3723, 1.3704, 0.0, 0.45, 33.67, 5.9867, 0.0, 0.0),
    ),
    (
        "ethylmethanoate",
        "ethylmethanoate",
        (1.3599, 1.3575, 0.0, 0.38, 33.36, 8.331, 0.0, 0.0),
    ),
    (
        "ethylphenylether",
        "ethylphenylether",
        (1.5076, 1.5254, 0.0, 0.32, 46.65, 4.1797, 0.667, 0.0),
    ),
    (
        "ethylbenzene",
        "ethylbenzene",
        (1.4959, 1.4932, 0.0, 0.15, 41.38, 2.4339, 0.75, 0.0),
    ),
    (
        "fluorobenzene",
        "fluorobenzene",
        (1.4684, 1.4629, 0.0, 0.1, 38.37, 5.42, 0.857, 0.143),
    ),
    ("formamide", "formamide", (1.4472, 1.4468, 0.62, 0.6, 82.08, 108.94, 0.0, 0.0)),
    ("formicacid", "formicacid", (1.3714, 1.3693, 0.75, 0.38, 53.44, 51.1, 0.0, 0.0)),
    ("n-heptane", "n-heptane", (1.3878, 1.3855, 0.0, 0.0, 28.28, 1.9113, 0.0, 0.0)),
    (
        "n-hexadecane",
        "n-hexadecane",
        (1.4345, 1.4325, 0.0, 0.0, 38.93, 2.0402, 0.0, 0.0),
    ),
    ("hexane", "n-hexane", (1.3749, 1.3722, 0.0, 0.0, 25.75, 1.8819, 0.0, 0.0)),
    ("hexanoicacid", "hexanoicacid", (1.4163, 1.4146, 0.6, 0.45, 39.65, 2.6, 0.0, 0.0)),
    ("iodobenzene", "iodobenzene", (1.62, 1.6172, 0.0, 0.12, 55.72, 4.547, 0.857, 0.0)),
    ("iodoethane", "iodoethane", (1.5133, 1.51, 0.0, 0.15, 40.96, 7.6177, 0.0, 0.0)),
    ("iodomethane", "iodomethane", (1.538, 1.527, 0.0, 0.13, 43.67, 6.865, 0.0, 0.0)),
    (
        "isopropylbenzene",
        "isopropylbenzene",
        (1.4915, 1.4889, 0.0, 0.16, 39.85, 2.3712, 0.667, 0.0),
    ),
    (
        "p-isopropyltoluene",
        "p-isopropyltoluene",
        (1.4909, 1.4885, 0.0, 0.19, 38.34, 2.2322, 0.6, 0.0),
    ),
    ("mesitylene", "mesitylene", (1.4994, 1.4968, 0.0, 0.19, 39.65, 2.265, 0.667, 0.0)),
    ("methanol", "methanol", (1.3288, 1.3265, 0.43, 0.47, 31.77, 32.613, 0.0, 0.0)),
    (
        "methylbenzoate",
        "methylbenzoate",
        (1.5164, 1.5146, 0.0, 0.46, 53.5, 6.7367, 0.6, 0.0),
    ),
    (
        "methylbutanoate",
        "methylbutanoate",
        (1.3878, 1.3847, 0.0, 0.45, 35.44, 5.5607, 0.0, 0.0),
    ),
    (
        "methylethanoate",
        "methylethanoate",
        (1.3614, 1.3589, 0.0, 0.45, 35.59, 6.8615, 0.0, 0.0),
    ),
    (
        "methylmethanoate",
        "methylmethanoate",
        (1.3433, 1.3415, 0.0, 0.38, 35.06, 8.8377, 0.0, 0.0),
    ),
    (
        "methylpropanoate",
        "methylpropanoate",
        (1.3775, 1.3742, 0.0, 0.45, 35.18, 6.0777, 0.0, 0.0),
    ),
    (
        "n-methylaniline",
        "N-methylaniline",
        (1.5684, 1.5681, 0.17, 0.43, 53.11, 5.96, 0.75, 0.0),
    ),
    (
        "methylcyclohexane",
        "methylcyclohexane",
        (1.4231, 1.4206, 0.0, 0.0, 33.52, 2.024, 0.0, 0.0),
    ),
    (
        "n-methylformamide-e-zmixture",
        "N-methylformamide(E/Zmixture)",
        (1.4319, 1.431, 0.4, 0.55, 55.44, 181.56, 0.0, 0.0),
    ),
    (
        "nitrobenzene",
        "nitrobenzene",
        (1.5562, 1.503, 0.0, 0.28, 57.54, 34.809, 0.667, 0.0),
    ),
    (
        "nitroethane",
        "nitroethane",
        (1.3917, 1.3897, 0.02, 0.33, 46.25, 28.29, 0.0, 0.0),
    ),
    (
        "nitromethane",
        "nitromethane",
        (1.3817, 1.3796, 0.06, 0.31, 52.58, 36.562, 0.0, 0.0),
    ),
    (
        "o-nitrotoluene",
        "o-nitrotoluene",
        (1.545, 1.5474, 0.0, 0.27, 59.12, 25.669, 0.6, 0.0),
    ),
    ("n-nonane", "n-nonane", (1.4054, 1.4031, 0.0, 0.0, 32.21, 1.9605, 0.0, 0.0)),
    ("n-octane", "n-octane", (1.3974, 1.3953, 0.0, 0.0, 30.43, 1.9406, 0.0, 0.0)),
    (
        "n-pentadecane",
        "n-pentadecane",
        (1.4315, 1.4298, 0.0, 0.0, 38.34, 2.0333, 0.0, 0.0),
    ),
    ("pentanal", "pentanal", (1.3944, 1.3917, 0.0, 0.4, 36.62, 10.0, 0.0, 0.0)),
    ("n-pentane", "n-pentane", (1.3575, 1.3547, 0.0, 0.0, 22.3, 1.8371, 0.0, 0.0)),
    (
        "pentanoic-acid",
        "pentanoic acid",
        (1.4085, 1.406, 0.6, 0.45, 38.4, 2.6924, 0.0, 0.0),
    ),
    (
        "pentyl-ethanoate",
        "pentyl ethanoate",
        (1.4023, -1.0, 0.0, 0.45, 36.23, 4.7297, 0.0, 0.0),
    ),
    ("pentylamine", "pentylamine", (1.448, 1.4093, 0.16, 0.61, 35.54, 4.201, 0.0, 0.0)),
    (
        "perfluorobenzene",
        "perfluorobenzene",
        (1.3777, 1.3761, 0.0, 0.0, 31.74, 2.029, 0.5, 0.5),
    ),
    ("propanal", "propanal", (1.3636, 1.3593, 0.0, 0.45, 32.48, 18.5, 0.0, 0.0)),
    (
        "propanoic-acid",
        "propanoic acid",
        (1.3869, 1.3848, 0.6, 0.45, 37.71, 3.44, 0.0, 0.0),
    ),
    (
        "propanonitrile",
        "propanonitrile",
        (1.3655, 1.3633, 0.02, 0.36, 38.5, 29.324, 0.0, 0.0),
    ),
    (
        "propyl-ethanoate",
        "propyl ethanoate",
        (1.3842, 1.3822, 0.0, 0.45, 34.26, 5.5205, 0.0, 0.0),
    ),
    (
        "propylamine",
        "propylamine",
        (1.387, 1.3851, 0.16, 0.61, 31.31, 4.9912, 0.0, 0.0),
    ),
    ("pyridine", "pyridine", (1.5095, 1.5073, 0.0, 0.52, 52.62, 12.978, 0.833, 0.0)),
    (
        "tetrachloroethene",
        "tetrachloroethene",
        (1.5053, 1.5055, 0.0, 0.0, 45.19, 2.268, 0.0, 0.667),
    ),
    (
        "tetrahydrofuran",
        "tetrahydrofuran",
        (1.405, 1.4044, 0.0, 0.48, 39.44, 7.4257, 0.0, 0.0),
    ),
    (
        "tetrahydrothiophene-s-s-dioxide",
        "tetrahydrothiophene-S,S-dioxide",
        (1.4833, -1.0, 0.0, 0.88, 87.49, 43.962, 0.0, 0.0),
    ),
    ("tetralin", "tetralin", (1.5413, 1.5392, 0.0, 0.19, 47.74, 2.771, 0.6, 0.0)),
    ("thiophene", "thiophene", (1.5289, 1.5268, 0.0, 0.15, 44.16, 2.727, 0.8, 0.0)),
    ("thiophenol", "thiophenol", (1.5893, 1.58, 0.09, 0.16, 55.24, 4.2728, 0.857, 0.0)),
    ("toluene", "toluene", (1.4961, 1.4936, 0.0, 0.14, 40.2, 2.3741, 0.857, 0.0)),
    (
        "trans-decalin",
        "trans-decalin",
        (1.4695, 1.4671, 0.0, 0.0, 42.19, 2.1781, 0.0, 0.0),
    ),
    (
        "tributylphosphate",
        "tributylphosphate",
        (1.4224, 1.4215, 0.0, 1.21, 27.55, 8.1781, 0.0, 0.0),
    ),
    (
        "trichloroethene",
        "trichloroethene",
        (1.4773, 1.4556, 0.08, 0.03, 41.45, 3.422, 0.0, 0.6),
    ),
    (
        "triethylamine",
        "triethylamine",
        (1.401, 1.398, 0.0, 0.79, 29.1, 2.3832, 0.0, 0.0),
    ),
    ("n-undecane", "n-undecane", (1.4398, 1.4151, 0.0, 0.0, 34.85, 1.991, 0.0, 0.0)),
    ("water", "water", (1.3328, 1.3323, 0.82, 0.35, -1.0, 78.355, -1.0, -1.0)),
    (
        "xylene-mixture",
        "xylene (mixture)",
        (1.4995, 1.4969, 0.0, 0.16, 41.38, 2.3879, 0.75, 0.0),
    ),
    ("m-xylene", "m-xylene", (1.4972, 1.4946, 0.0, 0.16, 40.98, 2.3478, 0.75, 0.0)),
    ("o-xylene", "o-xylene", (1.5055, 1.5029, 0.0, 0.16, 42.83, 2.5454, 0.75, 0.0)),
    ("p-xylene", "p-xylene", (1.4958, 1.4932, 0.0, 0.16, 40.32, 2.2705, 0.75, 0.0)),
)

__all__ = [
    "PYSCF_SMD_SOURCE_MODULE_SHA256",
    "PYSCF_SMD_SOURCE_VERSION",
    "PYSCF_SMD_TABLE_SHA256",
    "SMD_SOLVENT_DESCRIPTOR_SNAPSHOT",
]

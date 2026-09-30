"""Native PySCF 2.13.1 SMD-CDS parameter literal semantics.

The legacy Fortran stores unsuffixed ``DATA`` literals in implicit-double
arrays.  Those values are first rounded as default REAL (binary32) and only
then promoted to binary64.  Explicit ``D0`` values remain binary64.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import Mapping

from maple.function.route2_solvents import (
    Route2SolventSpec,
    SMDSolventDescriptors,
    route2_solvent_spec,
)

UPSTREAM_MNSOL_SHA256 = (
    "f57b94c0eb6d5a29f1a1441294795321a5a39af74e0bec890628fac83027250b"
)
PYSCF_BOHR_ANGSTROM = 0.52917721092
LEGACY_TOANGS = 0.52917724924
LEGACY_HARTREE_TO_KCAL_MOL = 627.509451
PROBE_RADIUS_ANGSTROM = 0.4
COORDINATE_SCALE = LEGACY_TOANGS / PYSCF_BOHR_ANGSTROM


def promote_fortran_real_literal(value: float) -> float:
    """Round one unsuffixed Fortran REAL literal, then promote to binary64."""

    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("Fortran REAL literal must be finite.")
    return float(struct.unpack("!f", struct.pack("!f", normalized))[0])


def _promoted(values: Mapping[object, float]) -> Mapping:
    return MappingProxyType(
        {key: promote_fortran_real_literal(value) for key, value in values.items()}
    )


BONDI_MANTINA_ANGSTROM = _promoted(
    {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 1.98,
    }
)

AQ_SIGMA = _promoted(
    {
        1: 48.69,
        6: 129.74,
        9: 38.18,
        16: -9.10,
        17: 9.82,
        35: -8.72,
        101: -72.95,
        103: 68.69,
        105: -48.22,
        106: 121.98,
        114: 68.85,
        116: 84.10,
    }
)
AQ_HSIGMA = _promoted({6: -60.77})
NAQ_SIGMA_N = _promoted(
    {
        6: 58.10,
        7: 32.62,
        8: -17.56,
        14: -18.04,
        16: -33.17,
        17: -24.31,
        35: -35.42,
        101: -62.05,
        103: -15.70,
        110: -99.76,
    }
)
NAQ_SIGMA_A = _promoted({6: 48.10, 8: 193.06, 103: 95.99, 105: -41.00, 110: 152.20})
NAQ_SIGMA_B = _promoted({6: 32.87, 8: -43.79, 104: -128.16, 106: 79.13})
NAQ_HSIGMA_N = _promoted({6: -36.37, 8: -19.39})
NAQ_HSIGMA_A = MappingProxyType({})
NAQ_HSIGMA_B = MappingProxyType({})
SIGMA_MOL = tuple(
    promote_fortran_real_literal(value) for value in (0.35, 0.0, -4.19, -6.68)
)

# Reachable D0 RKKVAL entries in the qualified H/C/O domain.  The reused
# atomic-tension kernel owns the complete table and its operation order.
REFERENCE_DISTANCE_ANGSTROM = MappingProxyType(
    {
        ("H", "C"): 1.55,
        ("H", "O"): 1.55,
        ("C", "H"): 1.55,
        ("C", "C"): 1.84,
        ("C", "O"): 1.84,
        ("O", "H"): 1.55,
        ("O", "C"): 1.33,
        ("O", "O"): 1.80,
    }
)

NATIVE_LITERAL_CLASS_MANIFEST = MappingProxyType(
    {
        "bondi": "binary32-promoted-to-binary64",
        "aq_sigma_hsigma": "binary32-promoted-to-binary64",
        "naq_sigma_hsigma": "binary32-promoted-to-binary64",
        "sigma_mol": "binary32-promoted-to-binary64",
        "probe_radius": "binary64-D0",
        "reference_distances": "binary64-D0",
        "coordinate_conversion": "binary64-D-exponent",
        "energy_conversion": "binary64-D-exponent",
        "dareal_pi_epsi": "binary64-D0",
    }
)


@dataclass(frozen=True, slots=True)
class NativeSMDParameterSet:
    """Immutable parameters after native Fortran literal promotion."""

    sigma: Mapping[int, float]
    hsigma: Mapping[int, float]
    cssigma_cal_mol_angstrom2: float
    molecular_terms_cal_mol_angstrom2: tuple[float, float, float, float]
    icds: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", MappingProxyType(dict(self.sigma)))
        object.__setattr__(self, "hsigma", MappingProxyType(dict(self.hsigma)))


def aqueous_parameters() -> NativeSMDParameterSet:
    return NativeSMDParameterSet(
        sigma=AQ_SIGMA,
        hsigma=AQ_HSIGMA,
        cssigma_cal_mol_angstrom2=0.0,
        molecular_terms_cal_mol_angstrom2=(0.0, 0.0, 0.0, 0.0),
        icds=1,
    )


def nonaqueous_parameters(
    descriptors: SMDSolventDescriptors,
) -> NativeSMDParameterSet:
    """Evaluate SMD_CDS_NAQ in the source expression order."""

    sigma_keys = set(NAQ_SIGMA_N) | set(NAQ_SIGMA_A) | set(NAQ_SIGMA_B)
    sigma = {
        key: (
            NAQ_SIGMA_N.get(key, 0.0) * descriptors.refractive_index
            + NAQ_SIGMA_A.get(key, 0.0) * descriptors.hydrogen_bond_acidity
            + NAQ_SIGMA_B.get(key, 0.0) * descriptors.hydrogen_bond_basicity
        )
        for key in sigma_keys
    }
    hsigma_keys = set(NAQ_HSIGMA_N) | set(NAQ_HSIGMA_A) | set(NAQ_HSIGMA_B)
    hsigma = {
        key: (
            NAQ_HSIGMA_N.get(key, 0.0) * descriptors.refractive_index
            + NAQ_HSIGMA_A.get(key, 0.0) * descriptors.hydrogen_bond_acidity
            + NAQ_HSIGMA_B.get(key, 0.0) * descriptors.hydrogen_bond_basicity
        )
        for key in hsigma_keys
    }
    terms = (
        SIGMA_MOL[0] * descriptors.surface_tension,
        SIGMA_MOL[1]
        * descriptors.hydrogen_bond_basicity
        * descriptors.hydrogen_bond_basicity,
        SIGMA_MOL[2]
        * descriptors.aromatic_carbon_fraction
        * descriptors.aromatic_carbon_fraction,
        SIGMA_MOL[3]
        * descriptors.electronegative_halogen_fraction
        * descriptors.electronegative_halogen_fraction,
    )
    cssigma = ((terms[0] + terms[1]) + terms[2]) + terms[3]
    return NativeSMDParameterSet(
        sigma=sigma,
        hsigma=hsigma,
        cssigma_cal_mol_angstrom2=cssigma,
        molecular_terms_cal_mol_angstrom2=terms,
        icds=2,
    )


def parameters_for_solvent(
    solvent: str,
) -> tuple[Route2SolventSpec, NativeSMDParameterSet]:
    spec = route2_solvent_spec(solvent)
    if spec.name not in {"water", "ethanol", "hexane"}:
        raise ValueError(
            "Native SMD CDS initially supports only water, ethanol, and hexane."
        )
    parameters = (
        aqueous_parameters()
        if spec.name == "water"
        else nonaqueous_parameters(spec.descriptors)
    )
    return spec, parameters


__all__ = [
    "AQ_HSIGMA",
    "AQ_SIGMA",
    "BONDI_MANTINA_ANGSTROM",
    "COORDINATE_SCALE",
    "LEGACY_HARTREE_TO_KCAL_MOL",
    "LEGACY_TOANGS",
    "NAQ_HSIGMA_A",
    "NAQ_HSIGMA_B",
    "NAQ_HSIGMA_N",
    "NAQ_SIGMA_A",
    "NAQ_SIGMA_B",
    "NAQ_SIGMA_N",
    "NATIVE_LITERAL_CLASS_MANIFEST",
    "NativeSMDParameterSet",
    "PROBE_RADIUS_ANGSTROM",
    "PYSCF_BOHR_ANGSTROM",
    "REFERENCE_DISTANCE_ANGSTROM",
    "SIGMA_MOL",
    "UPSTREAM_MNSOL_SHA256",
    "aqueous_parameters",
    "nonaqueous_parameters",
    "parameters_for_solvent",
    "promote_fortran_real_literal",
]

"""Exact Torch implementation of PySCF 2.13.1 legacy SMD CDS.

The parameter algebra and DAREAL surface definition are translated from
``pyscf/lib/solvent/mnsol.F`` (Apache-2.0), itself attributed there to the
NWChem Minnesota solvation implementation.  The scientific parameters are
the published SMD parameters and are intentionally not configurable here.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from maple.function.route2_solvents import route2_solvent_spec
from maple.solvation.coupling.operator import source_files_sha256
from maple.solvation.surfaces.legacy_dareal import (
    LegacyDAREALConfig,
    LegacyDAREALDiagnostics,
    legacy_dareal_areas_torch,
)

_PYSCF_BOHR_ANGSTROM = 0.52917721092
_LEGACY_TOANGS = 0.52917724924
_LEGACY_HARTREE_TO_KCAL_MOL = 627.509451
_HARTREE_TO_EV = 27.211386024367243
_COORDINATE_SCALE = _LEGACY_TOANGS / _PYSCF_BOHR_ANGSTROM
_SOURCE_SHA256 = "f57b94c0eb6d5a29f1a1441294795321a5a39af74e0bec890628fac83027250b"

_ATOMIC_NUMBER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
_BONDI_MANTINA = {
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

# Sparse 1-based SIGMA and HSIGMA tables from SMD_CDS_AQ/SMD_CDS_NAQ.
_AQ_SIGMA = MappingProxyType(
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
_AQ_HSIGMA = MappingProxyType({6: -60.77})
_NAQ_SIGMA_N = MappingProxyType(
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
_NAQ_SIGMA_A = MappingProxyType(
    {6: 48.10, 8: 193.06, 103: 95.99, 105: -41.00, 110: 152.20}
)
_NAQ_SIGMA_B = MappingProxyType({6: 32.87, 8: -43.79, 104: -128.16, 106: 79.13})
_NAQ_HSIGMA_N = MappingProxyType({6: -36.37, 8: -19.39})

# RKKVAL entries actually reachable for the supported element domain.
_REFERENCE_DISTANCE = {
    ("H", "C"): 1.55,
    ("H", "N"): 1.55,
    ("H", "O"): 1.55,
    ("H", "S"): 2.14,
    ("C", "H"): 1.55,
    ("C", "C"): 1.84,
    ("C", "N"): 1.84,
    ("C", "O"): 1.84,
    ("C", "F"): 1.84,
    ("C", "P"): 2.20,
    ("C", "S"): 2.20,
    ("C", "Cl"): 2.10,
    ("C", "Br"): 2.30,
    ("C", "I"): 2.60,
    ("N", "H"): 1.55,
    ("N", "C"): 1.84,
    ("N", "N"): 1.85,
    ("N", "O"): 1.50,
    ("O", "H"): 1.55,
    ("O", "C"): 1.33,
    ("O", "N"): 1.50,
    ("O", "O"): 1.80,
    ("O", "P"): 2.10,
    ("O", "S"): 1.71,
    ("F", "C"): 1.84,
    ("P", "C"): 2.20,
    ("P", "O"): 2.10,
    ("P", "S"): 2.50,
    ("S", "H"): 2.14,
    ("S", "C"): 2.20,
    ("S", "O"): 1.71,
    ("S", "S"): 2.75,
    ("S", "P"): 2.50,
    ("Cl", "C"): 2.10,
    ("Br", "C"): 2.30,
    ("I", "C"): 2.60,
}


@dataclass(frozen=True)
class LegacySMDCDSConfig:
    dareal: LegacyDAREALConfig = LegacyDAREALConfig()
    probe_radius_angstrom: float = 0.4
    implementation_version: str = "torch-legacy-smd-cds-pyscf-2.13.1-v1"


@dataclass(frozen=True)
class LegacySMDCDSDiagnostics:
    solvent: str
    icds: int
    dareal: LegacyDAREALDiagnostics
    coordinate_scale: float


@dataclass(frozen=True)
class LegacySMDCDSState:
    energy_eV: torch.Tensor
    energy_hartree: torch.Tensor
    energy_kcal_mol: torch.Tensor
    atom_areas_angstrom2: torch.Tensor
    atom_tensions_cal_mol_angstrom2: torch.Tensor
    diagnostics: LegacySMDCDSDiagnostics
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _host(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _switch(distance: torch.Tensor, reference: float, width: float) -> torch.Tensor:
    cutoff_delta = _host(distance) - (reference + width)
    if abs(cutoff_delta) <= 1.0e-11:
        raise RuntimeError(
            "Legacy SMD CDS coordination switch is on its cutoff boundary."
        )
    if cutoff_delta >= 0.0 or reference == 0.0:
        return distance.new_zeros(())
    return torch.exp(distance.new_tensor(width) / (distance - reference - width))


def _parameter_tables(solvent: str):
    spec = route2_solvent_spec(solvent)
    if spec.name == "water":
        return (
            MappingProxyType(dict(_AQ_SIGMA)),
            MappingProxyType(dict(_AQ_HSIGMA)),
            0.0,
            1,
        )
    d = spec.descriptors
    sigma_keys = set(_NAQ_SIGMA_N) | set(_NAQ_SIGMA_A) | set(_NAQ_SIGMA_B)
    sigma = {
        key: _NAQ_SIGMA_N.get(key, 0.0) * d.refractive_index
        + _NAQ_SIGMA_A.get(key, 0.0) * d.hydrogen_bond_acidity
        + _NAQ_SIGMA_B.get(key, 0.0) * d.hydrogen_bond_basicity
        for key in sigma_keys
    }
    hsigma = {key: value * d.refractive_index for key, value in _NAQ_HSIGMA_N.items()}
    cssigma = (
        0.35 * d.surface_tension
        - 4.19 * d.aromatic_carbon_fraction**2
        - 6.68 * d.electronegative_halogen_fraction**2
    )
    return MappingProxyType(sigma), MappingProxyType(hsigma), cssigma, 2


def _atomic_tensions(
    symbols: tuple[str, ...],
    positions: torch.Tensor,
    sigma: Mapping[int, float],
    hsigma: Mapping[int, float],
) -> torch.Tensor:
    n = len(symbols)
    distances: dict[tuple[int, int], torch.Tensor] = {}

    def distance(first: int, second: int) -> torch.Tensor:
        key = (min(first, second), max(first, second))
        if key not in distances:
            distances[key] = torch.linalg.vector_norm(
                positions[first] - positions[second]
            )
        return distances[key]

    cot: list[list[torch.Tensor]] = [
        [positions.new_zeros(()) for _ in range(n)] for _ in range(n)
    ]
    for i, first in enumerate(symbols):
        for j, second in enumerate(symbols):
            if i == j:
                continue
            reference = _REFERENCE_DISTANCE.get((first, second), 0.0)
            width = 0.10 if {first, second} == {"C", "O"} else 0.30
            if width == 0.10:
                reference = 1.33
            cot[i][j] = _switch(distance(i, j), reference, width)

    tensions: list[torch.Tensor] = []
    for i, symbol in enumerate(symbols):
        z = _ATOMIC_NUMBER[symbol]
        tension = positions.new_tensor(sigma.get(z, 0.0))
        if symbol == "H":
            for j, other in enumerate(symbols):
                tension = tension + cot[i][j] * hsigma.get(_ATOMIC_NUMBER[other], 0.0)
        elif symbol == "O":
            correction_index = {"C": 103, "N": 106, "S": 118, "P": 114}
            for j, other in enumerate(symbols):
                if other == "O" and i != j:
                    tension = tension + cot[i][j] * sigma.get(104, 0.0)
                elif other in correction_index:
                    tension = tension + cot[i][j] * sigma.get(
                        correction_index[other], 0.0
                    )
        elif symbol == "N":
            rtkk_sum = positions.new_zeros(())
            carbonyl_sum = positions.new_zeros(())
            for j, other in enumerate(symbols):
                if other != "C":
                    continue
                environment = positions.new_zeros(())
                oxygen_environment = positions.new_zeros(())
                for q, neighbor in enumerate(symbols):
                    if q in (i, j):
                        continue
                    value = _switch(
                        distance(j, q),
                        _REFERENCE_DISTANCE.get(("C", neighbor), 0.0),
                        0.30,
                    )
                    environment = environment + value
                    if neighbor == "O":
                        oxygen_environment = oxygen_environment + value
                rtkk_sum = rtkk_sum + cot[i][j] * environment**2
                carbonyl_sum = carbonyl_sum + cot[i][j] * oxygen_environment
            # The exact zero branch avoids undefined second derivatives of x**1.3 at x=0.
            if _host(rtkk_sum) != 0.0:
                tension = tension + rtkk_sum**1.3 * sigma.get(105, 0.0)
            tension = tension + carbonyl_sum * sigma.get(111, 0.0)
            triple = positions.new_zeros(())
            for j, other in enumerate(symbols):
                if i != j and other == "C":
                    triple = triple + _switch(distance(i, j), 1.225, 0.065)
            tension = tension + triple * sigma.get(116, 0.0)
        elif symbol == "C":
            cc1 = positions.new_zeros(())
            cc2 = positions.new_zeros(())
            cn = positions.new_zeros(())
            for j, other in enumerate(symbols):
                if i == j:
                    continue
                if other == "C":
                    cc1 = cc1 + _switch(distance(i, j), 1.84, 0.30)
                    cc2 = cc2 + _switch(distance(i, j), 1.27, 0.07)
                elif other == "N":
                    cn = cn + cot[i][j]
            tension = tension + cc1 * sigma.get(101, 0.0) + cc2 * sigma.get(102, 0.0)
            tension = tension + cn * cn * sigma.get(110, 0.0)
        elif symbol == "S":
            for j, other in enumerate(symbols):
                if i != j and other == "S":
                    tension = tension + cot[i][j] * sigma.get(107, 0.0)
                elif other == "P":
                    tension = tension + cot[i][j] * sigma.get(115, 0.0)
        tensions.append(tension)
    return torch.stack(tensions)


class TorchLegacySMDCDS:
    """Coordinate-connected legacy SMD CDS energy functional."""

    def __init__(
        self,
        symbols: Sequence[str],
        solvent: str,
        device: str | torch.device = "cpu",
        *,
        config: LegacySMDCDSConfig | None = None,
    ) -> None:
        normalized = tuple(str(symbol) for symbol in symbols)
        unsupported = sorted(set(normalized).difference(_ATOMIC_NUMBER))
        if not normalized:
            raise ValueError("Legacy SMD CDS requires at least one atom.")
        if unsupported:
            raise ValueError(
                f"Unsupported legacy SMD CDS elements: {', '.join(unsupported)}."
            )
        self._symbols = normalized
        self._solvent_spec = route2_solvent_spec(solvent)
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {self._device} is unavailable.")
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._config = LegacySMDCDSConfig() if config is None else config
        self._radii = torch.tensor(
            [
                _BONDI_MANTINA[symbol] + self._config.probe_radius_angstrom
                for symbol in normalized
            ],
            dtype=torch.float64,
            device=self._device,
        )
        self._sigma, self._hsigma, self._cssigma, self._icds = _parameter_tables(
            self._solvent_spec.name
        )
        module_path = Path(__file__).resolve()
        self._source_files_sha256 = source_files_sha256(
            {
                "maple.solvation.nonpolar.legacy_smd_cds": module_path,
                "maple.solvation.surfaces.legacy_dareal": (
                    module_path.parents[1] / "surfaces" / "legacy_dareal.py"
                ),
                "maple.function.route2_solvents": (
                    module_path.parents[2] / "function" / "route2_solvents.py"
                ),
            }
        )
        self._provenance = MappingProxyType(
            {
                "provider": "torch-legacy-smd-cds",
                "implementation_version": self._config.implementation_version,
                "scientific_model": "PySCF-2.13.1-get_cds_legacy",
                "upstream": "pyscf/lib/solvent/mnsol.F",
                "upstream_license": "Apache-2.0",
                "upstream_sha256": _SOURCE_SHA256,
                "solvent": self._solvent_spec.name,
                "descriptor_source": self._solvent_spec.descriptor_source,
                "source_files_sha256": json.dumps(
                    self._source_files_sha256, separators=(",", ":")
                ),
            }
        )

    @property
    def config(self) -> LegacySMDCDSConfig:
        return self._config

    @property
    def provenance(self) -> Mapping[str, str]:
        return self._provenance

    @property
    def provenance_sha256(self) -> str:
        payload = json.dumps(
            dict(self._provenance), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def configuration_sha256(self) -> str:
        payload = {
            "symbols": self._symbols,
            "solvent": self._solvent_spec.name,
            "device": str(self._device),
            "config": asdict(self._config),
            "radii_angstrom": tuple(
                float(value) for value in self._radii.detach().cpu()
            ),
            "upstream_sha256": _SOURCE_SHA256,
            "source_files_sha256": self._source_files_sha256,
            "sigma": tuple(
                sorted((int(key), float(value)) for key, value in self._sigma.items())
            ),
            "hsigma": tuple(
                sorted((int(key), float(value)) for key, value in self._hsigma.items())
            ),
            "cssigma": float(self._cssigma),
            "icds": int(self._icds),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def device(self) -> torch.device:
        return self._device

    def evaluate_torch(self, positions_angstrom: torch.Tensor) -> LegacySMDCDSState:
        if not isinstance(positions_angstrom, torch.Tensor):
            raise TypeError("Legacy SMD CDS positions must be a Torch tensor.")
        if positions_angstrom.dtype != torch.float64:
            raise TypeError("Legacy SMD CDS requires torch.float64 positions.")
        if positions_angstrom.device != self._device:
            raise ValueError(
                f"Positions are on {positions_angstrom.device}, expected {self._device}."
            )
        if positions_angstrom.shape != (len(self._symbols), 3):
            raise ValueError(
                f"Legacy SMD CDS positions must have shape {(len(self._symbols), 3)}."
            )
        if not bool(torch.isfinite(positions_angstrom).all()):
            raise ValueError("Legacy SMD CDS positions must be finite.")

        # Reproduce PySCF's Angstrom -> Bohr conversion followed by the old
        # Fortran TOANGS conversion, rather than cancelling them algebraically.
        legacy_positions = positions_angstrom * _COORDINATE_SCALE
        dareal = legacy_dareal_areas_torch(
            legacy_positions, self._radii, config=self._config.dareal
        )
        tensions = _atomic_tensions(
            self._symbols, legacy_positions, self._sigma, self._hsigma
        )
        energy_kcal = (
            torch.sum(dareal.areas_angstrom2 * (tensions + self._cssigma)) * 0.001
        )
        energy_hartree = energy_kcal / _LEGACY_HARTREE_TO_KCAL_MOL
        energy_ev = energy_hartree * _HARTREE_TO_EV
        if not bool(torch.isfinite(energy_ev)):
            raise RuntimeError("Legacy SMD CDS produced a non-finite energy.")
        return LegacySMDCDSState(
            energy_eV=energy_ev,
            energy_hartree=energy_hartree,
            energy_kcal_mol=energy_kcal,
            atom_areas_angstrom2=dareal.areas_angstrom2,
            atom_tensions_cal_mol_angstrom2=tensions,
            diagnostics=LegacySMDCDSDiagnostics(
                solvent=self._solvent_spec.name,
                icds=self._icds,
                dareal=dareal.diagnostics,
                coordinate_scale=_COORDINATE_SCALE,
            ),
            provenance=self._provenance,
        )

    def energy_torch(self, positions_angstrom: torch.Tensor) -> torch.Tensor:
        return self.evaluate_torch(positions_angstrom).energy_eV


__all__ = [
    "LegacySMDCDSConfig",
    "LegacySMDCDSDiagnostics",
    "LegacySMDCDSState",
    "TorchLegacySMDCDS",
]

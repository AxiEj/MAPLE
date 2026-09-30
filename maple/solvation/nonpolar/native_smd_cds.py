"""Native-literal Torch reproduction of PySCF 2.13.1 SMD-CDS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.operator import source_files_sha256
from maple.solvation.nonpolar.legacy_smd_cds import (
    LegacySMDCDSDiagnostics,
    LegacySMDCDSState,
    _atomic_tensions,
)
from maple.solvation.nonpolar.native_smd_cds_parameters import (
    BONDI_MANTINA_ANGSTROM,
    COORDINATE_SCALE,
    LEGACY_HARTREE_TO_KCAL_MOL,
    NATIVE_LITERAL_CLASS_MANIFEST,
    PROBE_RADIUS_ANGSTROM,
    UPSTREAM_MNSOL_SHA256,
    parameters_for_solvent,
)
from maple.solvation.surfaces.legacy_dareal import (
    LegacyDAREALConfig,
    legacy_dareal_areas_torch,
)

_SUPPORTED_ELEMENTS = frozenset({"H", "C", "O"})
_IMPLEMENTATION_VERSION = "torch-native-smd-cds-pyscf-2.13.1-v2"
_PROVIDER = "torch-native-smd-cds"


@dataclass(frozen=True, slots=True)
class NativeSMDCDSConfig:
    """Closed scientific configuration for the native-literal provider."""

    dareal: LegacyDAREALConfig = LegacyDAREALConfig()
    probe_radius_angstrom: float = PROBE_RADIUS_ANGSTROM
    implementation_version: str = _IMPLEMENTATION_VERSION


def _sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class TorchNativeSMDCDS:
    """Coordinate-connected native-literal SMD-CDS for H/C/O systems."""

    __slots__ = (
        "_config",
        "_cssigma",
        "_device",
        "_hsigma",
        "_icds",
        "_initial_configuration_sha256",
        "_molecular_terms",
        "_provenance",
        "_radii",
        "_sigma",
        "_solvent_spec",
        "_source_paths",
        "_symbols",
    )

    def __init__(
        self,
        symbols: Sequence[str],
        solvent: str,
        device: str | torch.device = "cpu",
    ) -> None:
        normalized = tuple(str(symbol) for symbol in symbols)
        if not normalized:
            raise ValueError("Native SMD CDS requires at least one atom.")
        unsupported = sorted(set(normalized).difference(_SUPPORTED_ELEMENTS))
        if unsupported:
            raise ValueError(
                "Native SMD CDS initially supports only H, C, and O; "
                f"received unsupported elements: {', '.join(unsupported)}."
            )
        self._symbols = normalized
        self._solvent_spec, parameters = parameters_for_solvent(solvent)
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {self._device} is unavailable.")
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())

        self._config = NativeSMDCDSConfig()
        self._radii = torch.tensor(
            [
                BONDI_MANTINA_ANGSTROM[symbol] + self._config.probe_radius_angstrom
                for symbol in normalized
            ],
            dtype=torch.float64,
            device=self._device,
        )
        self._sigma = parameters.sigma
        self._hsigma = parameters.hsigma
        self._cssigma = parameters.cssigma_cal_mol_angstrom2
        self._molecular_terms = parameters.molecular_terms_cal_mol_angstrom2
        self._icds = parameters.icds

        module_path = Path(__file__).resolve()
        self._source_paths = MappingProxyType(
            {
                "maple.function.route2_solvents": (
                    module_path.parents[2] / "function" / "route2_solvents.py"
                ),
                "maple.solvation.api.units": (
                    module_path.parents[1] / "api" / "units.py"
                ),
                "maple.solvation.nonpolar.legacy_smd_cds.atomic_tensions": (
                    module_path.with_name("legacy_smd_cds.py")
                ),
                "maple.solvation.nonpolar.native_smd_cds": module_path,
                "maple.solvation.nonpolar.native_smd_cds_parameters": (
                    module_path.with_name("native_smd_cds_parameters.py")
                ),
                "maple.solvation.surfaces.legacy_dareal": (
                    module_path.parents[1] / "surfaces" / "legacy_dareal.py"
                ),
            }
        )
        sources = self._current_source_files_sha256()
        self._provenance = MappingProxyType(
            {
                "provider": _PROVIDER,
                "implementation_version": _IMPLEMENTATION_VERSION,
                "scientific_model": "PySCF-2.13.1-get_cds_legacy-native-literals",
                "upstream": "pyscf/lib/solvent/mnsol.F",
                "upstream_license": "Apache-2.0",
                "upstream_sha256": UPSTREAM_MNSOL_SHA256,
                "solvent": self._solvent_spec.name,
                "descriptor_source": self._solvent_spec.descriptor_source,
                "literal_class_manifest": json.dumps(
                    dict(NATIVE_LITERAL_CLASS_MANIFEST),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "source_files_sha256": json.dumps(sources, separators=(",", ":")),
            }
        )
        self._initial_configuration_sha256 = self._current_configuration_sha256()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def provenance(self) -> Mapping[str, str]:
        return self._provenance

    @property
    def provenance_sha256(self) -> str:
        return _sha(dict(self._provenance))

    def _current_source_files_sha256(self) -> tuple[tuple[str, str], ...]:
        return source_files_sha256(dict(self._source_paths))

    def _current_configuration_sha256(self) -> str:
        payload = {
            "symbols": self._symbols,
            "solvent": self._solvent_spec.name,
            "device": str(self._device),
            "config": asdict(self._config),
            "radii_angstrom": tuple(
                float(value) for value in self._radii.detach().cpu()
            ),
            "sigma": tuple(
                sorted((int(key), float(value)) for key, value in self._sigma.items())
            ),
            "hsigma": tuple(
                sorted((int(key), float(value)) for key, value in self._hsigma.items())
            ),
            "cssigma": float(self._cssigma),
            "molecular_terms": tuple(float(value) for value in self._molecular_terms),
            "icds": int(self._icds),
            "coordinate_scale": float(COORDINATE_SCALE),
            "hartree_to_kcal_mol": float(LEGACY_HARTREE_TO_KCAL_MOL),
            "hartree_to_eV": float(HARTREE_TO_EV),
            "upstream_sha256": UPSTREAM_MNSOL_SHA256,
            "literal_class_manifest": tuple(
                sorted(NATIVE_LITERAL_CLASS_MANIFEST.items())
            ),
            "provenance": tuple(sorted(self._provenance.items())),
            "source_files_sha256": self._current_source_files_sha256(),
        }
        return _sha(payload)

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._initial_configuration_sha256:
            raise RuntimeError(
                "Native SMD CDS configuration mutated after construction."
            )
        return current

    def _validate_positions(self, positions_angstrom: torch.Tensor) -> None:
        if not isinstance(positions_angstrom, torch.Tensor):
            raise TypeError("Native SMD CDS positions must be a Torch tensor.")
        if positions_angstrom.dtype != torch.float64:
            raise TypeError("Native SMD CDS requires torch.float64 positions.")
        if positions_angstrom.device != self._device:
            raise ValueError(
                f"Positions are on {positions_angstrom.device}, expected {self._device}."
            )
        expected_shape = (len(self._symbols), 3)
        if positions_angstrom.shape != expected_shape:
            raise ValueError(
                f"Native SMD CDS positions must have shape {expected_shape}."
            )
        if not bool(torch.isfinite(positions_angstrom).all()):
            raise ValueError("Native SMD CDS positions must be finite.")

    def evaluate_torch(self, positions_angstrom: torch.Tensor) -> LegacySMDCDSState:
        self.configuration_sha256()
        self._validate_positions(positions_angstrom)
        try:
            legacy_positions = positions_angstrom * COORDINATE_SCALE
            dareal = legacy_dareal_areas_torch(
                legacy_positions, self._radii, config=self._config.dareal
            )
            tensions = _atomic_tensions(
                self._symbols, legacy_positions, self._sigma, self._hsigma
            )
            energy_kcal = (
                torch.sum(dareal.areas_angstrom2 * (tensions + self._cssigma)) * 0.001
            )
            energy_hartree = energy_kcal / LEGACY_HARTREE_TO_KCAL_MOL
            energy_ev = energy_hartree * HARTREE_TO_EV
            if not bool(torch.isfinite(energy_ev)):
                raise RuntimeError("Native SMD CDS produced a non-finite energy.")
            state = LegacySMDCDSState(
                energy_eV=energy_ev,
                energy_hartree=energy_hartree,
                energy_kcal_mol=energy_kcal,
                atom_areas_angstrom2=dareal.areas_angstrom2,
                atom_tensions_cal_mol_angstrom2=tensions,
                diagnostics=LegacySMDCDSDiagnostics(
                    solvent=self._solvent_spec.name,
                    icds=self._icds,
                    dareal=dareal.diagnostics,
                    coordinate_scale=COORDINATE_SCALE,
                ),
                provenance=self._provenance,
            )
        finally:
            self.configuration_sha256()
        return state

    def energy_torch(self, positions_angstrom: torch.Tensor) -> torch.Tensor:
        return self.evaluate_torch(positions_angstrom).energy_eV


__all__ = ["NativeSMDCDSConfig", "TorchNativeSMDCDS"]

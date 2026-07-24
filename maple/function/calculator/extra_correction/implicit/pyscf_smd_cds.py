"""Optional official PySCF SMD CDS energy and analytic coordinate gradient.

PySCF is imported lazily and remains an optional Route-2 research dependency.
The adapter calls the same compiled ``get_cds_legacy`` entrypoint used by
PySCF's production SMD energy and gradient paths; MAPLE does not reproduce or
modify the upstream CDS functional.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from ase.units import Bohr

from .pyscf_runtime import (
    TESTED_PYSCF_VERSION,
    require_tested_pyscf_version,
)
from .smd_cds import HARTREE_TO_KCAL_MOL, validate_smd_symbols


@dataclass(frozen=True)
class _PySCFSMDCDSRuntime:
    version: str
    gto: Any
    smd: Any


@dataclass(frozen=True)
class PySCFSMDCDSResult:
    energy_hartree: float
    energy_kcal_mol: float
    position_gradient_hartree_per_angstrom: np.ndarray
    runtime_provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        gradient = np.asarray(
            self.position_gradient_hartree_per_angstrom,
            dtype=float,
        ).copy()
        gradient.setflags(write=False)
        object.__setattr__(
            self,
            "position_gradient_hartree_per_angstrom",
            gradient,
        )
        object.__setattr__(
            self,
            "runtime_provenance",
            MappingProxyType(dict(self.runtime_provenance)),
        )


def _load_pyscf_smd_cds_runtime() -> _PySCFSMDCDSRuntime:
    try:
        pyscf = importlib.import_module("pyscf")
        gto = importlib.import_module("pyscf.gto")
        smd = importlib.import_module("pyscf.solvent.smd")
    except ImportError as exc:
        raise ImportError(
            "The optional PySCF SMD CDS research adapter requires "
            f"PySCF {TESTED_PYSCF_VERSION}; MAPLE does not install it "
            "automatically."
        ) from exc
    return _PySCFSMDCDSRuntime(
        version=str(pyscf.__version__),
        gto=gto,
        smd=smd,
    )


def pyscf_smd_water_cds(
    symbols,
    positions_angstrom: np.ndarray,
    *,
    _runtime: _PySCFSMDCDSRuntime | None = None,
) -> PySCFSMDCDSResult:
    """Return official PySCF water-SMD CDS energy and gradient.

    The upstream gradient is in hartree/bohr.  MAPLE returns the position
    gradient, not force, in hartree/angstrom.
    """

    runtime = (
        _load_pyscf_smd_cds_runtime()
        if _runtime is None
        else _runtime
    )
    version = require_tested_pyscf_version(
        runtime.version,
        feature="The optional PySCF SMD CDS bridge",
    )
    if getattr(runtime.smd, "libsolvent", None) is None:
        raise RuntimeError(
            "The optional PySCF SMD CDS bridge requires PySCF's compiled "
            "libsolvent implementation."
        )

    normalized_symbols = validate_smd_symbols(tuple(symbols))
    if not normalized_symbols:
        raise ValueError("PySCF SMD CDS requires at least one atom.")
    positions = np.asarray(positions_angstrom, dtype=float)
    expected_shape = (len(normalized_symbols), 3)
    if positions.shape != expected_shape or not np.all(np.isfinite(positions)):
        raise ValueError(
            "PySCF SMD CDS coordinates must be finite with shape "
            f"{expected_shape}; received {positions.shape}."
        )

    molecule = runtime.gto.M(
        atom=list(
            zip(
                normalized_symbols,
                positions.tolist(),
                strict=True,
            )
        ),
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )
    smd_object = runtime.smd.SMD(molecule, solvent="water")
    energy, gradient_bohr = runtime.smd.get_cds_legacy(smd_object)

    energy_hartree = float(energy)
    if not np.isfinite(energy_hartree):
        raise RuntimeError("PySCF SMD CDS energy must be finite.")
    gradient_bohr = np.asarray(gradient_bohr, dtype=float)
    if (
        gradient_bohr.shape != expected_shape
        or not np.all(np.isfinite(gradient_bohr))
    ):
        raise RuntimeError(
            "PySCF SMD CDS gradient must be finite with shape "
            f"{expected_shape}; received {gradient_bohr.shape}."
        )

    return PySCFSMDCDSResult(
        energy_hartree=energy_hartree,
        energy_kcal_mol=energy_hartree * HARTREE_TO_KCAL_MOL,
        position_gradient_hartree_per_angstrom=(
            gradient_bohr / Bohr
        ),
        runtime_provenance={
            "provider": "pyscf-smd-libsolvent-cds",
            "pyscf_version": version,
            "solvent": "water",
            "upstream_entrypoint": "pyscf.solvent.smd.get_cds_legacy",
        },
    )

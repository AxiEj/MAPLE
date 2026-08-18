"""One-shot AIMNet2 fixed-charge ddPCM plus SMD-CDS comparator factory.

This module binds the source-reconstructed CPU float64 AIMNet2 checkpoint to
the water harmonic-ddPCM electrostatic candidate and the official PySCF-2.13.1
SMD-CDS energy/gradient companion.  AIMNet2 is evaluated once per geometry;
the continuum field is never supplied to the model and there is no electronic
fixed point.

The electrostatic term is a fixed-Hirshfeld-like-point-charge continuum
baseline, not original density-based self-consistent SMD.  ``SMD`` remains in
the historical Python entry-point names for compatibility and refers only to
the separately evaluated PySCF SMD-CDS component and solvent parameter table.
The factory creates the exact composite scalar and its guarded ASE E/F bridge,
but does not by itself make a profile publicly selectable.
"""

from __future__ import annotations

from pathlib import Path

import torch

from maple.function.calculator.aimnet._aimnet2_float64_source import (
    AIMNet2ReconstructedFloat64SourceCalculator,
)
from maple.solvation.api import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_PROFILE_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
)
from maple.solvation.continuum.harmonic_point_ddpcm_torch_functional import (
    build_water_aimnet2_frozen_charge_harmonic_ddpcm_candidate,
)
from maple.solvation.continuum.aimnet2_smooth_partition_ddpcm import (
    build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate,
)
from maple.solvation.coupling.geometry_mediated import (
    GeometryMediatedElectrostaticScalar,
)
from maple.solvation.coupling.geometry_mediated_ase import (
    GeometryMediatedScalarASECalculator,
)
from maple.solvation.coupling.geometry_mediated_smd import (
    GeometryMediatedMultisolventSMDTotalScalar,
    GeometryMediatedSMDTotalScalar,
    PySCFMultisolventSMDCDSNonpolarFunctional,
    PySCFSMDCDSNonpolarFunctional,
)
from maple.solvation.models.aimnet2 import (
    AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT,
    AIMNET2_WB97M_D3_FROZEN_CHARGE_WATER_FLOAT64_CONTRACT,
    AIMNet2GeometryMediatedModelAdapter,
)


def _verify_checkpoint(checkpoint: Path) -> None:
    contract = AIMNET2_WB97M_D3_FROZEN_CHARGE_WATER_FLOAT64_CONTRACT
    if (
        checkpoint.stat().st_size != contract.checkpoint_size_bytes
        or _sha256_file(checkpoint) != contract.checkpoint_sha256
    ):
        raise ValueError(
            "The AIMNet2 checkpoint bytes do not match the frozen-charge "
            "water CPU-float64 Route-2 contract."
        )


def _verify_multisolvent_checkpoint(checkpoint: Path) -> None:
    contract = AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT
    if (
        checkpoint.stat().st_size != contract.checkpoint_size_bytes
        or _sha256_file(checkpoint) != contract.checkpoint_sha256
    ):
        raise ValueError(
            "The AIMNet2 checkpoint bytes do not match the frozen-charge "
            "multi-solvent CPU-float64 Route-2 contract."
        )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_aimnet2_frozen_charge_water_smd_scalar(
    atoms,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> GeometryMediatedSMDTotalScalar:
    """Build the exact aqueous total scalar for one fixed atomic identity."""

    if device != "cpu":
        raise ValueError(
            "The admitted source-reconstructed AIMNet2 float64 runtime is CPU-only."
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    _verify_checkpoint(checkpoint_path)
    source_calculator = AIMNet2ReconstructedFloat64SourceCalculator(
        model_path=checkpoint_path,
        device=device,
    )
    model = AIMNet2GeometryMediatedModelAdapter(
        source_calculator,
        contract=AIMNET2_WB97M_D3_FROZEN_CHARGE_WATER_FLOAT64_CONTRACT,
    )
    model.domain.validate_atoms(atoms)
    symbols_getter = getattr(atoms, "get_chemical_symbols", None)
    if not callable(symbols_getter):
        raise TypeError("AIMNet2 frozen-charge SMD geometry requires symbols.")
    continuum = build_water_aimnet2_frozen_charge_harmonic_ddpcm_candidate(
        tuple(str(symbol) for symbol in symbols_getter()),
        dtype=torch.float64,
        device=device,
    )
    electrostatic = GeometryMediatedElectrostaticScalar(
        model,
        continuum,
        scalar_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
        ),
        profile_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
        ),
    )
    return GeometryMediatedSMDTotalScalar(
        electrostatic,
        PySCFSMDCDSNonpolarFunctional(),
    )


def build_aimnet2_frozen_charge_water_smd_ase_calculator(
    atoms,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> GeometryMediatedScalarASECalculator:
    """Build the guarded ASE energy/free-energy/forces bridge."""

    return GeometryMediatedScalarASECalculator(
        build_aimnet2_frozen_charge_water_smd_scalar(
            atoms,
            checkpoint,
            device=device,
        )
    )


def build_aimnet2_frozen_charge_smooth_partition_smd_scalar(
    atoms,
    checkpoint: str | Path,
    *,
    solvent: str,
    device: str = "cpu",
) -> GeometryMediatedMultisolventSMDTotalScalar:
    """Build the disabled high-fidelity multi-solvent one-shot total scalar."""

    if device != "cpu":
        raise ValueError(
            "The admitted source-reconstructed AIMNet2 float64 runtime is CPU-only."
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    _verify_multisolvent_checkpoint(checkpoint_path)
    source_calculator = AIMNet2ReconstructedFloat64SourceCalculator(
        model_path=checkpoint_path,
        device=device,
    )
    model = AIMNet2GeometryMediatedModelAdapter(
        source_calculator,
        contract=AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT,
    )
    model.domain.validate_atoms(atoms)
    symbols_getter = getattr(atoms, "get_chemical_symbols", None)
    if not callable(symbols_getter):
        raise TypeError("AIMNet2 frozen-charge SMD geometry requires symbols.")
    continuum = build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate(
        tuple(str(symbol) for symbol in symbols_getter()),
        solvent=solvent,
        dtype=torch.float64,
        device=device,
    )
    electrostatic = GeometryMediatedElectrostaticScalar(
        model,
        continuum,
        scalar_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1
        ),
        profile_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_PROFILE_V1
        ),
    )
    return GeometryMediatedMultisolventSMDTotalScalar(
        electrostatic,
        PySCFMultisolventSMDCDSNonpolarFunctional(solvent=continuum.solvent),
    )


def build_aimnet2_frozen_charge_smooth_partition_smd_ase_calculator(
    atoms,
    checkpoint: str | Path,
    *,
    solvent: str,
    device: str = "cpu",
) -> GeometryMediatedScalarASECalculator:
    """Build the guarded ASE E/F bridge for the disabled multi-solvent scalar."""

    return GeometryMediatedScalarASECalculator(
        build_aimnet2_frozen_charge_smooth_partition_smd_scalar(
            atoms,
            checkpoint,
            solvent=solvent,
            device=device,
        )
    )


__all__ = [
    "build_aimnet2_frozen_charge_smooth_partition_smd_ase_calculator",
    "build_aimnet2_frozen_charge_smooth_partition_smd_scalar",
    "build_aimnet2_frozen_charge_water_smd_ase_calculator",
    "build_aimnet2_frozen_charge_water_smd_scalar",
]

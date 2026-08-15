"""Optional, source-independent scientific reference builders.

This package is not imported by the production Route-2 runtime.  Its modules
keep optional QM and native continuum dependencies behind explicit call sites
so dependency-light MAPLE imports remain unchanged.
"""

from maple.solvation.reference.pyscf_pcmsolver import (
    AOInverseDistanceIntegralCache,
    PCMSolverDensityResponse,
    PCMSolverSCFSolvent,
    array_sha256,
    attach_pcmsolver_to_scf,
    closed_shell_density_from_orbitals,
    response_symmetry_defect,
    solvent_energy_directional_derivative_error,
)

__all__ = [
    "AOInverseDistanceIntegralCache",
    "PCMSolverDensityResponse",
    "PCMSolverSCFSolvent",
    "array_sha256",
    "attach_pcmsolver_to_scf",
    "closed_shell_density_from_orbitals",
    "response_symmetry_defect",
    "solvent_energy_directional_derivative_error",
]

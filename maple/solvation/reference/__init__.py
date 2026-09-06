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
from maple.solvation.reference.auxiliary_density import (
    AuxiliaryDensityProjection,
    coulomb_fit_auxiliary_density,
)
from maple.solvation.reference.exterior_probe_partition import (
    DENSE_EXTERIOR_AUDIT_ROTATION,
    DENSE_EXTERIOR_LEBEDEV_ORDER,
    DENSE_EXTERIOR_LEBEDEV_POINTS,
    DENSE_EXTERIOR_PARTITION_NAMES,
    DenseExteriorProbePartition,
    build_dense_exterior_probe_partition,
)
from maple.solvation.reference.radial_gto_observable_span import (
    RadialGTOObservableSpanResult,
    evaluate_radial_gto_observable_span,
)
from maple.solvation.reference.sector_balanced_response_modes import (
    SectorBalancedResponseModes,
    select_sector_balanced_response_modes,
)
from maple.solvation.reference.permanent_quadrupole_probes import (
    PERMANENT_P13_LEBEDEV_ORDER,
    PERMANENT_P13_LEBEDEV_POINTS,
    build_permanent_p13_probe_partition,
)
from maple.solvation.reference.p13_permanent_span import (
    P13PermanentSpanResult,
    evaluate_p13_permanent_span,
)

__all__ = [
    "AOInverseDistanceIntegralCache",
    "AuxiliaryDensityProjection",
    "DENSE_EXTERIOR_AUDIT_ROTATION",
    "DENSE_EXTERIOR_LEBEDEV_ORDER",
    "DENSE_EXTERIOR_LEBEDEV_POINTS",
    "DENSE_EXTERIOR_PARTITION_NAMES",
    "DenseExteriorProbePartition",
    "PCMSolverDensityResponse",
    "PCMSolverSCFSolvent",
    "PERMANENT_P13_LEBEDEV_ORDER",
    "PERMANENT_P13_LEBEDEV_POINTS",
    "P13PermanentSpanResult",
    "RadialGTOObservableSpanResult",
    "SectorBalancedResponseModes",
    "array_sha256",
    "attach_pcmsolver_to_scf",
    "closed_shell_density_from_orbitals",
    "coulomb_fit_auxiliary_density",
    "build_dense_exterior_probe_partition",
    "build_permanent_p13_probe_partition",
    "evaluate_radial_gto_observable_span",
    "select_sector_balanced_response_modes",
    "evaluate_p13_permanent_span",
    "response_symmetry_defect",
    "solvent_energy_directional_derivative_error",
]

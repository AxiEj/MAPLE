"""Release-evidence helpers for the conservative Route-2 rebuild.

The release package never admits a scientific capability.  It only captures
and validates source/runtime identity plus raw diagnostic measurements used by
the authoritative registries.
"""

from .evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    sha256_file,
    write_external_json_artifact,
)
from .pes_validation import (
    closed_loop_work,
    closed_rectangular_loop,
    displace_positions,
    reverse_closed_path,
    summarize_directional_derivatives,
    water_geometry_descriptors,
    water_vibrational_directions,
)
from .box_convergence import (
    BOX_LENGTHS_A,
    MAXIMUM_ADJOINT_RESIDUAL,
    MAXIMUM_PRIMAL_RESIDUAL,
    TAIL_CONTINUUM_ENERGY_TOLERANCE_EV,
    TAIL_FORCE_MAX_TOLERANCE_EV_PER_A,
    TAIL_FORCE_RMS_TOLERANCE_EV_PER_A,
    TAIL_SOURCE_RELATIVE_TOLERANCE,
    TAIL_TOTAL_ENERGY_TOLERANCE_EV,
    summarize_box_convergence,
)

__all__ = [
    "RepositorySnapshot",
    "canonical_json_sha256",
    "checkpoint_record",
    "collect_loaded_repository_sources",
    "committed_source_hashes",
    "runtime_record",
    "sha256_file",
    "write_external_json_artifact",
    "closed_loop_work",
    "closed_rectangular_loop",
    "displace_positions",
    "reverse_closed_path",
    "summarize_directional_derivatives",
    "water_geometry_descriptors",
    "water_vibrational_directions",
    "BOX_LENGTHS_A",
    "MAXIMUM_ADJOINT_RESIDUAL",
    "MAXIMUM_PRIMAL_RESIDUAL",
    "TAIL_CONTINUUM_ENERGY_TOLERANCE_EV",
    "TAIL_FORCE_MAX_TOLERANCE_EV_PER_A",
    "TAIL_FORCE_RMS_TOLERANCE_EV_PER_A",
    "TAIL_SOURCE_RELATIVE_TOLERANCE",
    "TAIL_TOTAL_ENERGY_TOLERANCE_EV",
    "summarize_box_convergence",
]

"""Backward-compatible re-export shim for the MD utilities.

The former single ``utils`` module mixed ~9 concerns; it has been split into
focused, single-concern sibling modules.  This module re-exports every name that
``utils`` previously exposed so existing ``from ...md.utils import X`` call sites
keep working unchanged.  The concern modules below are the source of truth;
import from them directly when a module wants an explicit, narrow dependency:

* :mod:`.units`             — physical constants, unit conversions, ``forces_au``
* :mod:`.pbc`               — image flags, wrap/unwrap reconstruction
* :mod:`.capabilities`      — calc-capability gates, stress contract, admission
* :mod:`.thermo`            — temperature, kinetic energy
* :mod:`.dof`               — DOF policy helpers, linear-molecule test
* :mod:`.pressure`          — instantaneous (virial) pressure
* :mod:`.velocity_init`     — Maxwell-Boltzmann init, temperature rescaling
* :mod:`.motion_projection` — COM/angular projection, velocity reprs, momenta
* :mod:`.trajectory_xyz`    — extended-XYZ frame writers
"""

from .units import (  # noqa: F401
    AMU_ANG2_PER_FS2_TO_EV,
    AMU_TO_AU,
    ANGSTROM_TO_BOHR,
    AU_TO_FS,
    BAR_TO_EV_PER_ANG3,
    BOHR_TO_ANGSTROM,
    DEFAULT_COMPRESSIBILITY,
    EV_PER_ANG3_TO_BAR,
    EV_PER_ANG_TO_AU,
    EV_TO_HARTREE,
    FS_TO_AU,
    HA_PER_ANG_TO_AU,
    HARTREE_TO_EV,
    HARTREE_TO_KELVIN,
    KELVIN_TO_HARTREE,
    VELOCITY_REPR_LFMIDDLE_CARRIED,
    VELOCITY_REPR_STANDARD,
    _VALID_VELOCITY_REPRESENTATIONS,
    forces_au,
)
from .pbc import (  # noqa: F401
    IMAGE_FLAGS_ARRAY,
    WRAP_BOUNDARY_EPS,
    copy_with_unwrapped_positions,
    ensure_image_flags,
    get_unwrapped_positions,
    wrap_positions_with_image_flags,
)
from .capabilities import (  # noqa: F401
    MDStressUnavailableError,
    _calc_capability,
    _calc_label,
    _calc_model_name,
    pbc_com_default_note,
    validate_md_capabilities,
    validate_md_parameter_ranges,
    validate_stress_tensor,
)
from .thermo import calculate_kinetic_energy, calculate_temperature  # noqa: F401
from .dof import (  # noqa: F401
    describe_dof_policy,
    get_initialization_dof_policy,
    get_n_dof_from_policy,
    get_runtime_dof_policy,
    is_linear_molecule,
)
from .pressure import compute_instantaneous_pressure  # noqa: F401
from .velocity_init import (  # noqa: F401
    initialize_velocities,
    scale_velocities_to_temperature,
)
from .motion_projection import (  # noqa: F401
    apply_runtime_motion_projection,
    calculate_angular_momentum,
    calculate_momentum,
    get_atoms_velocity_representation,
    lfmiddle_carried_to_standard,
    normalize_velocity_representation,
    remove_center_of_mass_motion,
    remove_rigid_body_rotation,
    set_atoms_velocity_representation,
    standard_to_lfmiddle_carried,
)
from .trajectory_xyz import write_xyz_frame, write_xyz_trajectory  # noqa: F401


__all__ = [
    # units
    "AMU_ANG2_PER_FS2_TO_EV", "AMU_TO_AU", "ANGSTROM_TO_BOHR", "AU_TO_FS",
    "BAR_TO_EV_PER_ANG3", "BOHR_TO_ANGSTROM", "DEFAULT_COMPRESSIBILITY",
    "EV_PER_ANG3_TO_BAR", "EV_PER_ANG_TO_AU", "EV_TO_HARTREE", "FS_TO_AU",
    "HA_PER_ANG_TO_AU", "HARTREE_TO_EV", "HARTREE_TO_KELVIN", "KELVIN_TO_HARTREE",
    "VELOCITY_REPR_LFMIDDLE_CARRIED", "VELOCITY_REPR_STANDARD",
    "_VALID_VELOCITY_REPRESENTATIONS", "forces_au",
    # pbc
    "IMAGE_FLAGS_ARRAY", "WRAP_BOUNDARY_EPS", "copy_with_unwrapped_positions",
    "ensure_image_flags", "get_unwrapped_positions", "wrap_positions_with_image_flags",
    # capabilities
    "MDStressUnavailableError", "_calc_capability", "_calc_label", "_calc_model_name",
    "pbc_com_default_note", "validate_md_capabilities", "validate_md_parameter_ranges",
    "validate_stress_tensor",
    # thermo
    "calculate_kinetic_energy", "calculate_temperature",
    # dof
    "describe_dof_policy", "get_initialization_dof_policy", "get_n_dof_from_policy",
    "get_runtime_dof_policy", "is_linear_molecule",
    # pressure
    "compute_instantaneous_pressure",
    # velocity_init
    "initialize_velocities", "scale_velocities_to_temperature",
    # motion_projection
    "apply_runtime_motion_projection", "calculate_angular_momentum", "calculate_momentum",
    "get_atoms_velocity_representation", "lfmiddle_carried_to_standard",
    "normalize_velocity_representation", "remove_center_of_mass_motion",
    "remove_rigid_body_rotation", "set_atoms_velocity_representation",
    "standard_to_lfmiddle_carried",
    # trajectory_xyz
    "write_xyz_frame", "write_xyz_trajectory",
]

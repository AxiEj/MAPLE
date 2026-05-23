"""WS5 — utils.py is a pure re-export shim over the split concern modules.

Guards that every name the former monolithic ``utils`` exposed is still
importable from ``...md.utils`` (so no call site breaks) and that the shim
re-exports the *same objects* defined in the concern modules.
"""

from maple.function.dispatcher.md import (
    capabilities,
    dof,
    motion_projection,
    pbc,
    pressure,
    semantics,
    thermo,
    trajectory_xyz,
    units,
    utils,
    velocity_init,
)


EXPECTED_NAMES = [
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
    "normalize_velocities_to_standard", "normalize_velocity_representation",
    "remove_center_of_mass_motion",
    "remove_rigid_body_rotation", "set_atoms_velocity_representation",
    "standard_to_lfmiddle_carried",
    # trajectory_xyz
    "write_xyz_frame", "write_xyz_trajectory",
    # semantics
    "validate_md_admission_state",
]


def test_every_previously_public_name_is_importable_from_utils():
    missing = [name for name in EXPECTED_NAMES if not hasattr(utils, name)]
    assert not missing, f"names dropped by the utils shim: {missing}"


def test_underscored_calc_helpers_importable_for_set_calculator():
    # set_calculator imports these underscored helpers from md.utils by name.
    from maple.function.dispatcher.md.utils import (
        _calc_capability,
        _calc_label,
        _calc_model_name,
    )
    assert callable(_calc_capability) and callable(_calc_label) and callable(_calc_model_name)


def test_shim_reexports_are_the_concern_module_objects():
    assert utils.forces_au is units.forces_au
    assert utils.HA_PER_ANG_TO_AU is units.HA_PER_ANG_TO_AU
    assert utils.wrap_positions_with_image_flags is pbc.wrap_positions_with_image_flags
    assert utils.validate_stress_tensor is capabilities.validate_stress_tensor
    assert utils.calculate_temperature is thermo.calculate_temperature
    assert utils.get_n_dof_from_policy is dof.get_n_dof_from_policy
    assert utils.compute_instantaneous_pressure is pressure.compute_instantaneous_pressure
    assert utils.initialize_velocities is velocity_init.initialize_velocities
    assert utils.apply_runtime_motion_projection is motion_projection.apply_runtime_motion_projection
    assert utils.write_xyz_frame is trajectory_xyz.write_xyz_frame
    assert utils.validate_md_admission_state is semantics.validate_md_admission_state


def test_all_is_complete():
    assert set(utils.__all__) == set(EXPECTED_NAMES)

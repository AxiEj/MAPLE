"""Public extension surface for compatible atomwise-potential MACE-EF models."""

from .extra_correction.implicit.mace_polar_ef_specs import (
    MACE_POLAR_EF_V2_CHECKPOINT_SPEC,
    MACEPolarEFCheckpointSpec,
    mace_polar_ef_checkpoint_spec,
    register_mace_polar_ef_checkpoint_spec,
    registered_mace_polar_ef_checkpoint_specs,
)
from .mace._macepolef_calculator import MACEPolarEFCalculator

__all__ = [
    "MACE_POLAR_EF_V2_CHECKPOINT_SPEC",
    "MACEPolarEFCalculator",
    "MACEPolarEFCheckpointSpec",
    "mace_polar_ef_checkpoint_spec",
    "register_mace_polar_ef_checkpoint_spec",
    "registered_mace_polar_ef_checkpoint_specs",
]

"""Route-2 coupling contracts: spaces, coordinates, pairing, and adjoints."""

from .metrics import ATOMIC_L1_PAIRING, AUTHORITATIVE_Q, PairingMetric, QPairing
from .operator import AdjointValidation, CouplingOperator, validate_adjoint_dot_product, validate_coupling_adjoint
from .spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE, ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates, AtomicL1FieldDualSpace, AtomicL1SourceSpace,
    ChargeConstrainedCoordinates, FieldDualSpace, SourceSpace,
)

__all__ = [
    "ATOMIC_L1_FIELD_DUAL_SPACE", "ATOMIC_L1_PAIRING", "ATOMIC_L1_SOURCE_SPACE",
    "AUTHORITATIVE_Q", "AdjointValidation", "AffineChargeCoordinates",
    "AtomicL1FieldDualSpace", "AtomicL1SourceSpace", "ChargeConstrainedCoordinates",
    "CouplingOperator", "FieldDualSpace", "PairingMetric", "QPairing", "SourceSpace",
    "validate_adjoint_dot_product", "validate_coupling_adjoint",
]

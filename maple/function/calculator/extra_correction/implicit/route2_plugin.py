"""Public facade for the capability-layered Route-2 MLIP plug-in contract."""

from .route2_mace_plugin import MACEPolarPlugin
from .route2_mace_p0 import (
    MACEP0AuditCertificate,
    MACE_P0_ARTIFACT_ID,
    born_point_charge_sign_canary,
    load_mace_p0_certificate,
    select_uniform_field_convention,
)
from .route2_plugin_contracts import (
    DifferentiableElectronicModel,
    ElectronicSourceProvider,
    FieldResponsiveModel,
    P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE,
    PluginAdmission,
    PluginAdmissionRoute,
    PluginCapabilityDeclaration,
    PluginElectronicState,
    PluginExecutionMode,
    PluginProfile,
    PluginProvenance,
    VariationalElectronicModel,
    admit_plugin,
)
from .route2_plugin_errors import PluginContractError
from .route2_plugin_legacy import (
    AtomicL1PluginEngineAdapter,
    LegacyAtomicL1BridgeSpec,
    adapt_atomic_l1_plugin_to_legacy_engine,
)
from .route2_plugin_registry import (
    ArtifactDispositionRecord,
    ArtifactDispositionRegistry,
    Route2PluginRegistry,
)
from .route2_plugin_spaces import (
    ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE,
    ATOMIC_L1_PLUGIN_SOURCE_SPACE,
    ContinuumCoupling,
    CouplingAdjointEvidence,
    ElectronicSourceSpace,
    FieldDualSpace,
)
from .route2_p1_audit import (
    FIELD_CONJUGATE_OPERATIONAL_PREDICTION,
    P1_AUDIT_VERSION,
    RESPONSE_CONDITIONED_OPERATIONAL_PREDICTION,
    ContinuumSolveEvidence,
    EnergySourceConjugacyEvidence,
    OuterSCFEvidence,
    Route2OperationalEnergyAudit,
    audit_outer_scf,
    audit_plugin_energy_source_conjugacy,
    build_operational_energy_audit,
    classify_continuum_solve_evidence,
)
from .route2_pminus1 import (
    P_MINUS_1_WATER_CONTINUUM_SPEC,
    P_MINUS_1_WATER_CONTINUUM_SPEC_ID,
    ElectrostaticsOnlyCDSResult,
    PMinus1WaterContinuumSpec,
)

DEFAULT_ROUTE2_PLUGIN_REGISTRY = Route2PluginRegistry()
DEFAULT_ROUTE2_PLUGIN_REGISTRY.register("mace-polar-1", MACEPolarPlugin.from_calculator)

__all__ = [
    "ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE",
    "ATOMIC_L1_PLUGIN_SOURCE_SPACE",
    "ArtifactDispositionRecord",
    "ArtifactDispositionRegistry",
    "AtomicL1PluginEngineAdapter",
    "ContinuumCoupling",
    "CouplingAdjointEvidence",
    "ContinuumSolveEvidence",
    "DEFAULT_ROUTE2_PLUGIN_REGISTRY",
    "DifferentiableElectronicModel",
    "ElectronicSourceProvider",
    "ElectronicSourceSpace",
    "EnergySourceConjugacyEvidence",
    "ElectrostaticsOnlyCDSResult",
    "FIELD_CONJUGATE_OPERATIONAL_PREDICTION",
    "FieldDualSpace",
    "FieldResponsiveModel",
    "MACEPolarPlugin",
    "MACEP0AuditCertificate",
    "MACE_P0_ARTIFACT_ID",
    "LegacyAtomicL1BridgeSpec",
    "P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE",
    "P_MINUS_1_WATER_CONTINUUM_SPEC",
    "P_MINUS_1_WATER_CONTINUUM_SPEC_ID",
    "P1_AUDIT_VERSION",
    "PMinus1WaterContinuumSpec",
    "PluginAdmission",
    "PluginAdmissionRoute",
    "PluginCapabilityDeclaration",
    "PluginContractError",
    "PluginElectronicState",
    "PluginExecutionMode",
    "PluginProfile",
    "PluginProvenance",
    "RESPONSE_CONDITIONED_OPERATIONAL_PREDICTION",
    "OuterSCFEvidence",
    "Route2OperationalEnergyAudit",
    "Route2PluginRegistry",
    "VariationalElectronicModel",
    "adapt_atomic_l1_plugin_to_legacy_engine",
    "admit_plugin",
    "audit_outer_scf",
    "audit_plugin_energy_source_conjugacy",
    "born_point_charge_sign_canary",
    "build_operational_energy_audit",
    "classify_continuum_solve_evidence",
    "load_mace_p0_certificate",
    "select_uniform_field_convention",
]

"""Disabled scalar-first MACE-POLAR variational-effective-source candidate.

This adapter deliberately changes the model source.  The original checkpoint
``density_coefficients`` remain available only through the separate operational
adapter.  Here the complete eight-channel source is the exact derivative of an
anchored field-energy scalar on the registered fixed-charge/gauge-reduced chart.

No E/F/H/V/M capability is admitted by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
    MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID,
)
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT,
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.operator import source_files_sha256
from maple.solvation.coupling.state_equation import provider_behavior_sha256
from .base import (
    ModelProvenance,
    atom_count,
    model_charge_and_multiplicity,
)
from .field_energy import FieldEnergyFunctional, GaugeReducedDualityMap, TorchGeometry
from .mace_polar import MACEPolarRadialGTOModelAdapter

MACE_POLAR_VARIATIONAL_FIELD_ENERGY_PROVIDER_ID = (
    "maple.route2.model.mace-polar-field-energy-anchored.impl.v1"
)
MACE_POLAR_VARIATIONAL_DUALITY_MAP_ID = (
    "maple.route2.duality.mace-polar-radial-gto-fixed-charge.v1"
)

MACE_POLAR_VARIATIONAL_DUALITY_MAP = GaugeReducedDualityMap(
    duality_map_id=MACE_POLAR_VARIATIONAL_DUALITY_MAP_ID,
    source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
    field_space=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    pairing_metric=MACE_POLAR_RADIAL_GTO_PAIRING,
    coordinate_contract=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT,
    conjugacy_sign=1,
)


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _variational_source_files(calculator: object) -> tuple[tuple[str, str], ...]:
    """Bind every locally executed scalar/coordinate implementation by bytes."""

    files: dict[str, Path] = {
        "maple.solvation.models.field_energy": Path(__file__).with_name(
            "field_energy.py"
        ),
        "maple.solvation.models.mace_polar_variational": Path(__file__),
    }
    calculator_source = getattr(calculator, "__class__", type(calculator))
    try:
        import inspect

        raw_calculator = inspect.getsourcefile(calculator_source)
        evaluator = getattr(calculator, "_long_range_evaluator", None)
        raw_evaluator = (
            None if evaluator is None else inspect.getsourcefile(type(evaluator))
        )
    except (TypeError, OSError):
        raw_calculator = None
        raw_evaluator = None
    if raw_calculator is not None:
        path = Path(raw_calculator).resolve()
        if path.is_file():
            files["bound_mace_polar_calculator"] = path
    if raw_evaluator is not None:
        path = Path(raw_evaluator).resolve()
        if path.is_file():
            files["bound_mace_polar_long_range_evaluator"] = path
    return source_files_sha256(files)


class MACEPolarDifferentiableFieldGraph:
    """Graph-preserving candidate-only view of the frozen calculator inference.

    The accepted legacy calculator remains byte-identical to its P0 certificate.
    This wrapper injects an equal-valued coordinate tensor only inside the new,
    disabled model identity, while preserving the evaluator's identity or
    center-of-mass coordinate policy.
    """

    __slots__ = ("_calculator", "_configuration_sha256", "_sealed")

    def __init__(self, calculator: object) -> None:
        for name in ("_batch_dict", "_model_forward"):
            if not callable(getattr(calculator, name, None)):
                raise TypeError(f"field graph requires calculator.{name}().")
        projector = getattr(calculator, "_reaction_projector", None)
        if not callable(getattr(projector, "use_model_field_features", None)):
            raise TypeError("field graph requires a model-feature context manager.")
        evaluator = getattr(calculator, "_long_range_evaluator", None)
        if not isinstance(getattr(evaluator, "is_default", None), bool):
            raise TypeError("field graph requires one registered evaluator policy.")
        object.__setattr__(self, "_calculator", calculator)
        object.__setattr__(
            self,
            "_configuration_sha256",
            _hash(
                {
                    "schema": "route2-mace-polar-differentiable-field-graph-v1",
                    "implementation_sha256": _implementation_sha256(),
                    "source_files_sha256": dict(_variational_source_files(calculator)),
                    "evaluator_profile": getattr(evaluator, "profile", None),
                    "coordinate_policy": (
                        "identity" if evaluator.is_default else "subtract-atom-mean"
                    ),
                }
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarDifferentiableFieldGraph is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        evaluator = self._calculator._long_range_evaluator
        return _hash(
            {
                "schema": "route2-mace-polar-differentiable-field-graph-v1",
                "implementation_sha256": _implementation_sha256(),
                "source_files_sha256": dict(
                    _variational_source_files(self._calculator)
                ),
                "evaluator_profile": getattr(evaluator, "profile", None),
                "coordinate_policy": (
                    "identity" if evaluator.is_default else "subtract-atom-mean"
                ),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise ValueError("MACE-POLAR differentiable field graph drifted.")
        return current

    def __call__(
        self,
        atoms: object,
        *,
        model_field_features,
        positions_angstrom,
    ) -> Mapping[str, object]:
        self.configuration_sha256()
        torch = __import__("torch")
        count = atom_count(atoms)
        if (
            not torch.is_tensor(model_field_features)
            or model_field_features.shape != (count, 8)
            or not torch.is_floating_point(model_field_features)
            or not bool(torch.isfinite(model_field_features).all())
        ):
            raise ValueError(
                "model_field_features must be finite floating point with shape (N,8)."
            )
        if (
            not torch.is_tensor(positions_angstrom)
            or positions_angstrom.shape != (count, 3)
            or not torch.is_floating_point(positions_angstrom)
            or not bool(torch.isfinite(positions_angstrom).all())
        ):
            raise ValueError(
                "positions_angstrom must be finite floating point with shape (N,3)."
            )
        dtype = getattr(self._calculator, "dtype")
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(torch, str(dtype).replace("torch.", ""), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("MACE-POLAR field graph dtype is not a Torch dtype.")
        positions = positions_angstrom.to(
            dtype=dtype,
            device=getattr(self._calculator, "device"),
        )
        reference = torch.as_tensor(
            atoms.get_positions(),
            dtype=positions.dtype,
            device=positions.device,
        )
        if not torch.equal(positions.detach(), reference):
            raise ValueError(
                "positions_angstrom must exactly represent the supplied atoms geometry."
            )
        evaluator = self._calculator._long_range_evaluator
        evaluated_positions = (
            positions
            if evaluator.is_default
            else positions - torch.mean(positions, dim=0)
        )
        batch = self._calculator._batch_dict(atoms)
        batch["positions"] = evaluated_positions
        with self._calculator._reaction_projector.use_model_field_features(
            model_field_features
        ):
            output = self._calculator._model_forward(
                batch,
                compute_force=False,
                compute_stress=False,
                compute_hessian=False,
            )
        if not isinstance(output, Mapping):
            raise TypeError("MACE-POLAR field graph must return one mapping.")
        return output


class MACEPolarVariationalFieldEnergy(FieldEnergyFunctional):
    """Anchored checkpoint field energy; all derivatives are base-class AD."""

    __slots__ = (
        "_base",
        "_configuration_sha256",
        "_field_graph",
        "capabilities",
        "coordinate_frame_policy",
        "coupling_id",
        "device",
        "domain",
        "dtype",
        "field_convention",
        "model_profile_id",
        "provenance",
        "provenance_sha256",
        "provider_id",
        "variational_functional_admitted",
    )
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE

    def __init__(
        self,
        base: MACEPolarRadialGTOModelAdapter,
        *,
        field_graph=None,
    ) -> None:
        if not isinstance(base, MACEPolarRadialGTOModelAdapter):
            raise TypeError("base must be MACEPolarRadialGTOModelAdapter.")
        base.configuration_sha256()
        calculator = base._calculator
        if field_graph is None:
            field_graph = MACEPolarDifferentiableFieldGraph(calculator)
        if not callable(field_graph) or not callable(
            getattr(field_graph, "configuration_sha256", None)
        ):
            raise TypeError(
                "variational adapter requires one content-addressed callable field graph."
            )
        required = ("polar_state", "route2_gto_field_projection_spec")
        for name in required:
            if not callable(getattr(calculator, name, None)):
                raise TypeError(f"variational adapter requires callable {name}().")
        inference_sha = _hash(
            {
                "implementation_sha256": _implementation_sha256(),
                "source_files_sha256": dict(
                    _variational_source_files(base._calculator)
                ),
                "base_provenance_sha256": base.provenance_sha256,
                "duality_map_sha256": MACE_POLAR_VARIATIONAL_DUALITY_MAP.configuration_sha256(),
                "field_transform_sha256": base.field_transform.configuration_sha256(),
                "field_graph_configuration_sha256": (
                    field_graph.configuration_sha256()
                ),
                "field_graph_behavior_sha256": provider_behavior_sha256(
                    field_graph,
                    ("__call__",),
                    label="mace_polar_variational_field_graph",
                ),
                "construction": "vacuum-energy-plus-zero-field-source-anchor-v1",
            }
        )
        provenance = ModelProvenance(
            provider_id=MACE_POLAR_VARIATIONAL_FIELD_ENERGY_PROVIDER_ID,
            model_profile_id=MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID,
            model_family="MACE-POLAR-1-field-energy-conjugate-effective-source",
            checkpoint_sha256=base.provenance.checkpoint_sha256,
            upstream_version=base.provenance.upstream_version,
            upstream_commit=base.provenance.upstream_commit,
            inference_code_sha256=inference_sha,
            dtype=base.dtype,
            device=base.device,
            domain=base.domain,
            field_convention=self.field_space.field_convention,
            coordinate_frame_policy=base.coordinate_frame_policy,
            optimizer_parameter_groups_audited=False,
            optimizer_audit_evidence_sha256=None,
        )
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_field_graph", field_graph)
        object.__setattr__(self, "provider_id", provenance.provider_id)
        object.__setattr__(self, "model_profile_id", provenance.model_profile_id)
        object.__setattr__(self, "coupling_id", MACE_POLAR_RADIAL_GTO_COUPLING_ID)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "provenance_sha256", provenance.sha256)
        object.__setattr__(self, "dtype", provenance.dtype)
        object.__setattr__(self, "device", provenance.device)
        object.__setattr__(self, "domain", provenance.domain)
        object.__setattr__(self, "field_convention", provenance.field_convention)
        object.__setattr__(
            self, "coordinate_frame_policy", provenance.coordinate_frame_policy
        )
        object.__setattr__(self, "capabilities", CapabilityStatus())
        object.__setattr__(self, "variational_functional_admitted", False)
        super().__init__(
            duality_map=MACE_POLAR_VARIATIONAL_DUALITY_MAP,
            dtype=getattr(calculator, "dtype"),
            device=getattr(calculator, "device"),
        )
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )

    @property
    def original_density_observable(self) -> MACEPolarRadialGTOModelAdapter:
        """Separate diagnostic provider; never the source returned by this model."""

        return self._base

    def _current_configuration_sha256(self) -> str:
        return _hash(
            {
                "schema": "route2-mace-polar-variational-field-energy-v1",
                "implementation_sha256": _implementation_sha256(),
                "source_files_sha256": dict(
                    _variational_source_files(self._base._calculator)
                ),
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "provenance_sha256": self.provenance_sha256,
                "base_configuration_sha256": self._base.configuration_sha256(),
                "base_provenance_sha256": self._base.provenance_sha256,
                "base_configuration_sha256_rechecked": (
                    self._base.configuration_sha256()
                ),
                "calculator_behavior_sha256": provider_behavior_sha256(
                    self._base._calculator,
                    ("route2_gto_field_projection_spec",),
                    label="mace_polar_variational_calculator",
                ),
                "field_graph_identity": {
                    "module": getattr(self._field_graph, "__module__", ""),
                    "qualname": getattr(
                        self._field_graph,
                        "__qualname__",
                        type(self._field_graph).__qualname__,
                    ),
                },
                "field_graph_configuration_sha256": (
                    self._field_graph.configuration_sha256()
                ),
                "field_graph_behavior_sha256": provider_behavior_sha256(
                    self._field_graph,
                    ("__call__",),
                    label="mace_polar_variational_field_graph",
                ),
                "duality_map_sha256": self.duality_map.configuration_sha256(),
                "field_transform_sha256": self._base.field_transform.configuration_sha256(),
                "coordinate_contract_id": MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
                "capabilities": self.capabilities.enabled_tiers,
                "variational_functional_admitted": False,
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise ValueError("MACE-POLAR variational adapter configuration drifted.")
        return current

    def _checkpoint_output_torch(self, geometry: TorchGeometry, reduced_field):
        atoms = geometry.atoms
        radial_field = self.duality_map.lift_reduced_field_torch(
            reduced_field,
            atom_count=atom_count(atoms),
            total_charge=float(model_charge_and_multiplicity(atoms)[0]),
        )
        matrix = radial_field.new_tensor(self._base.field_transform.matrix)
        feature_tensor = radial_field @ matrix.T
        output = self._field_graph(
            atoms,
            model_field_features=feature_tensor,
            positions_angstrom=geometry.positions,
        )
        if not isinstance(output, Mapping):
            raise TypeError("MACE-POLAR field graph must return one mapping.")
        return output

    def _intrinsic_energy_torch(self, geometry: TorchGeometry, reduced_field):
        output = self._checkpoint_output_torch(geometry, reduced_field)
        return self._energy_from_output(output)

    @staticmethod
    def _energy_from_output(output):
        torch = __import__("torch")
        energy = output.get("energy")
        if energy is None or not torch.is_tensor(energy):
            raise RuntimeError("MACE-POLAR omitted intrinsic energy.")
        if not bool(torch.isfinite(energy).all()):
            raise RuntimeError("MACE-POLAR intrinsic energy is non-finite.")
        return energy.sum()

    def _zero_field_anchor(self, geometry: TorchGeometry, like):
        torch = __import__("torch")
        atom_count_value = geometry.positions.shape[0]
        total_charge = float(model_charge_and_multiplicity(geometry.atoms)[0])
        coordinates = self.duality_map.coordinates(
            atom_count=atom_count_value, total_charge=total_charge
        )
        zero = torch.zeros(
            coordinates.reduced_dimension,
            dtype=like.dtype,
            device=like.device,
            requires_grad=True,
        )
        zero_output = self._checkpoint_output_torch(geometry, zero)
        zero_energy = self._energy_from_output(zero_output)
        (zero_gradient,) = torch.autograd.grad(
            zero_energy,
            (zero,),
            create_graph=bool(geometry.positions.requires_grad),
            allow_unused=False,
        )
        density = zero_output.get("density_coefficients")
        if density is None or density.shape != (atom_count_value, 4):
            raise RuntimeError("MACE-POLAR omitted the zero-field density anchor.")
        original_tensor = torch.zeros(
            (atom_count_value, 8), dtype=like.dtype, device=like.device
        )
        original_tensor[:, (0, 2, 3, 4)] = density
        charge_weights = torch.as_tensor(
            self.source_space.effective_charge_weights,
            dtype=like.dtype,
            device=like.device,
        )
        anchor_charge = torch.sum(original_tensor * charge_weights)
        charge_error = abs(float(anchor_charge.detach().cpu()) - total_charge)
        if charge_error > 2.0e-10:
            raise RuntimeError(
                "MACE-POLAR zero-field density anchor violates fixed total charge."
            )
        original_reduced = self.duality_map._tplus_apply_torch(
            original_tensor,
            atom_count=atom_count_value,
            total_charge=total_charge,
        )
        return zero_gradient, original_reduced

    def _energy_torch(self, geometry: TorchGeometry, reduced_field):
        intrinsic = self._intrinsic_energy_torch(geometry, reduced_field)
        zero_gradient, original_reduced = self._zero_field_anchor(
            geometry, reduced_field
        )
        # E_anc = E_vac + m0.xi + eps(xi)-eps(0)-eps'(0).xi.
        # Here E_vac==eps(0), so simplify the identical terms before numerical
        # evaluation instead of subtracting two roughly 2 keV quantities.
        return intrinsic + __import__("torch").dot(
            original_reduced - zero_gradient,
            reduced_field,
        )

    def evaluate_energy(
        self,
        atoms: object,
        field: object,
        *,
        gauge_potential: float = 0.0,
    ) -> float:
        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        total_charge = float(model_charge_and_multiplicity(atoms)[0])
        reduced, detected_gauge = self.duality_map.decompose_field(
            field, atom_count=atom_count(atoms), total_charge=total_charge
        )
        if abs(detected_gauge - gauge_potential) > 2.0e-12:
            raise ValueError("declared gauge potential does not match the input field.")
        return self.energy_eV(
            atoms,
            reduced,
            total_charge=total_charge,
            gauge_potential=gauge_potential,
        )

    def evaluate_source(self, atoms: object, field: object) -> np.ndarray:
        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        total_charge = float(model_charge_and_multiplicity(atoms)[0])
        reduced, _ = self.duality_map.decompose_field(
            field, atom_count=atom_count(atoms), total_charge=total_charge
        )
        source = self.source_from_energy(atoms, reduced, total_charge=total_charge)
        actual_charge = self.source_space.total_charge(
            source, atom_count=atom_count(atoms)
        )
        if not np.isclose(actual_charge, total_charge, atol=2.0e-12, rtol=0.0):
            raise RuntimeError("energy-gradient source violates fixed total charge.")
        return source

    def source_jvp_full(
        self, atoms: object, field: object, field_direction: object
    ) -> np.ndarray:
        total_charge = float(model_charge_and_multiplicity(atoms)[0])
        count = atom_count(atoms)
        reduced, _ = self.duality_map.decompose_field(
            field, atom_count=count, total_charge=total_charge
        )
        reduced_direction = self.duality_map.reduce_field(
            field_direction, atom_count=count, total_charge=total_charge
        )
        return self.source_jvp(
            atoms,
            reduced,
            reduced_direction,
            total_charge=total_charge,
        )

    def source_vjp_full(
        self, atoms: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        total_charge = float(model_charge_and_multiplicity(atoms)[0])
        count = atom_count(atoms)
        reduced, _ = self.duality_map.decompose_field(
            field, atom_count=count, total_charge=total_charge
        )
        reduced_cotangent = self.source_vjp(
            atoms,
            reduced,
            source_cotangent,
            total_charge=total_charge,
        )
        return self.duality_map.field_cotangent_from_reduced(
            reduced_cotangent,
            atom_count=count,
            total_charge=total_charge,
        )

    def metadata(self) -> dict[str, object]:
        release_contract = self._base.release_contract
        return {
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "configuration_sha256": self.configuration_sha256(),
            "provenance_sha256": self.provenance_sha256,
            "duality_map_sha256": self.duality_map.configuration_sha256(),
            "source_definition": "complete-eight-channel-energy-gradient-effective-source",
            "original_density_coefficients_role": "diagnostic-observable-only",
            "long_range_symmetry_contract_id": (
                release_contract.long_range_symmetry_contract_id
            ),
            "structural_so3_equivariance_admitted": (
                release_contract.structural_so3_equivariance_admitted
            ),
            "long_range_symmetry_claim_boundary": (
                release_contract.long_range_symmetry_claim_boundary
            ),
            "capabilities": {tier: False for tier in "EFHVM"},
        }


def build_mace_polar_variational_field_energy(
    base: MACEPolarRadialGTOModelAdapter,
    *,
    field_graph=None,
) -> MACEPolarVariationalFieldEnergy:
    return MACEPolarVariationalFieldEnergy(base, field_graph=field_graph)


__all__ = [
    "MACEPolarVariationalFieldEnergy",
    "MACEPolarDifferentiableFieldGraph",
    "MACE_POLAR_VARIATIONAL_DUALITY_MAP",
    "MACE_POLAR_VARIATIONAL_DUALITY_MAP_ID",
    "MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID",
    "MACE_POLAR_VARIATIONAL_FIELD_ENERGY_PROVIDER_ID",
    "build_mace_polar_variational_field_energy",
]

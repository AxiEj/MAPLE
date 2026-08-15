"""Finite-dielectric smooth harmonic ddPCM scalar for AIMNet2 charges.

This disabled research provider reuses the conductor candidate's smooth
weighted basis, Coulomb single layer, analytic point-source map, SO(3)
intertwiners, and sealed derivative machinery.  It adds the finite-dielectric
double-layer equation required by PCM; it is not a scaled COSMO energy.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from maple.solvation.api.profiles import (
    SMOOTH_HARMONIC_GALERKIN_DDPCM_CONFIGURATION_CONTRACT_ID,
    SMOOTH_HARMONIC_GALERKIN_DDPCM_CONTINUUM_PROFILE_ID,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
)

from .harmonic_point_torch_functional import (
    SmoothPointChargeHarmonicGalerkinFunctionalCandidate,
)
from .harmonic_torch_functional import (
    _MAXIMUM_REFERENCE_CONDITION_NUMBER,
    _positive_float,
)
from .harmonic_torch_primitives import _assemble_double_layer, _torch

SMOOTH_POINT_HARMONIC_DDPCM_PROVIDER_ID = (
    "maple.route2.continuum.smooth-harmonic-point-l0-ddpcm-functional.impl.v1"
)
SMOOTH_POINT_HARMONIC_DDPCM_FUNCTIONAL_CONTRACT_ID = (
    "maple.route2.continuum.smooth-harmonic-point-l0-ddpcm-same-scalar.v1"
)
DDX_EQUATION_REFERENCE_COMMIT = "4d79e3d9caeae5e602683572a71cb550414f9b09"
_RESIDUAL_THRESHOLD = 1.0e-10
_COTANGENT_CLOSURE_THRESHOLD = 1.0e-10


class _DDPCMAssembly(NamedTuple):
    weighted_basis: Any
    raw_single_layer: Any
    raw_double_layer: Any
    raw_source: Any
    mass: Any
    surface_operator: Any
    double_layer: Any
    source_operator: Any
    conductor_limit_operator: Any
    dielectric_operator: Any


class SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
    SmoothPointChargeHarmonicGalerkinFunctionalCandidate
):
    """Fixed-dimensional finite-dielectric ddPCM Galerkin diagnostic."""

    __slots__ = ("_dielectric",)

    provider_id = SMOOTH_POINT_HARMONIC_DDPCM_PROVIDER_ID
    functional_contract_id = SMOOTH_POINT_HARMONIC_DDPCM_FUNCTIONAL_CONTRACT_ID
    continuum_profile_id = SMOOTH_HARMONIC_GALERKIN_DDPCM_CONTINUUM_PROFILE_ID
    configuration_contract_id = SMOOTH_HARMONIC_GALERKIN_DDPCM_CONFIGURATION_CONTRACT_ID
    _accepted_scalar_ids = frozenset(
        {DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1}
    )
    _source_implementation_files = (
        "harmonic_point_source.py",
        "harmonic_point_torch_functional.py",
        "harmonic_point_ddpcm_torch_functional.py",
    )
    finite_dielectric_parameterization = True
    conductor_reference_only = False
    _geometry_quadrature = (
        "finite-band-exact exposure/point source plus invariant pair-axis "
        "Gauss-Legendre double layer"
    )

    def __init__(
        self,
        *,
        atomic_numbers: tuple[int, ...],
        radii_angstrom: tuple[float, ...],
        dielectric: float,
        transition_width_angstrom2: float,
        surface_lmax: int,
        exposure_lmax: int,
        exposure_radial_quadrature_order: int = 96,
        green_radial_quadrature_order: int = 128,
        dtype: object,
        device: object,
        scalar_id: str = (
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
        ),
    ) -> None:
        epsilon = _positive_float(dielectric, name="dielectric")
        if epsilon <= 1.0:
            raise ValueError("dielectric must be finite and greater than one.")
        object.__setattr__(self, "_dielectric", epsilon)
        super().__init__(
            atomic_numbers=atomic_numbers,
            radii_angstrom=radii_angstrom,
            transition_width_angstrom2=transition_width_angstrom2,
            surface_lmax=surface_lmax,
            exposure_lmax=exposure_lmax,
            exposure_radial_quadrature_order=exposure_radial_quadrature_order,
            green_radial_quadrature_order=green_radial_quadrature_order,
            dtype=dtype,
            device=device,
            scalar_id=scalar_id,
        )

    @property
    def dielectric(self) -> float:
        return self._dielectric

    @property
    def dielectric_jump_factor(self) -> float:
        return (self._dielectric + 1.0) / (self._dielectric - 1.0)

    def _configuration_extensions(self) -> dict[str, object]:
        return {
            "continuum_model": "finite-dielectric-ief-pcm/ddpcm-equations",
            "dielectric": self._dielectric,
            "dielectric_jump_factor": self.dielectric_jump_factor,
            "double_layer_convention": (
                "outward-source-normal Laplace principal value; "
                "self spectrum -2*pi/(2*l+1)"
            ),
            "ddx_equation_reference_commit": DDX_EQUATION_REFERENCE_COMMIT,
            "uniform_cosmo_dielectric_energy_scaling": False,
            "admission_identity": (
                "parameterized-diagnostic-only; any admitted solvent profile "
                "must bind dielectric and the full continuum configuration"
            ),
            "double_layer_quadrature": (
                "pair-axis split Gauss-Legendre with exact retained-band "
                "azimuthal contraction"
            ),
            "provider_field_semantics": (
                "energy-conjugate source derivative from the transpose/KKT chain; "
                "not the generally nonsymmetric primal apparent-charge response"
            ),
        }

    def _assembly_formula(self) -> str:
        return (
            "M=E.T E; A=E.T K E; D=E.T D0 E; S=E.T V; b=Sc; "
            "Mf=b; [2*pi*g(eps)M-D]phi=[2*pi*M-D]f; Ax=Mphi; "
            "G=-1/2 b.T x"
        )

    def _stationary_scalar_formula(self) -> str:
        return "-1/2 b^T x subject to Mf=b, R_eps phi=R_inf f, Ax=Mphi"

    def _geometry_assembly_label(self) -> str:
        return "same-scalar-E-K-D0-V-finite-dielectric-ddpcm"

    def _runtime_provenance_extensions(self) -> dict[str, object]:
        return {
            "continuum_model": "finite-dielectric-ddpcm",
            "dielectric": self._dielectric,
            "finite_dielectric_parameterization": True,
            "uniform_cosmo_dielectric_energy_scaling": False,
            "ddx_equation_reference_commit": DDX_EQUATION_REFERENCE_COMMIT,
            "double_layer_radial_quadrature_order": self._green_radial_order,
            "admission_identity": (
                "parameterized-diagnostic-only; solvent-bound profile required"
            ),
            "provider_field_semantics": (
                "energy-conjugate source derivative from the transpose/KKT chain"
            ),
        }

    @staticmethod
    def _condition_number(operator: Any) -> float:
        return float(_torch().linalg.cond(operator).detach().cpu())

    def _assemble_ddpcm_torch(self, positions: Any) -> _DDPCMAssembly:
        (
            weighted_basis,
            raw_single_layer,
            raw_source,
            surface,
            source_operator,
        ) = super()._assemble_torch(positions)
        raw_double_layer = _assemble_double_layer(
            positions,
            radii=self._radii_angstrom,
            lmax=self.physical_lmax,
            radial_order=self._green_radial_order,
        )
        mass = weighted_basis.T @ weighted_basis
        mass = 0.5 * (mass + mass.T)
        double_layer = weighted_basis.T @ raw_double_layer @ weighted_basis
        conductor_limit = 2.0 * np.pi * mass - double_layer
        dielectric_operator = (
            2.0 * np.pi * self.dielectric_jump_factor * mass - double_layer
        )
        conditions = {
            "mass": self._condition_number(mass),
            "surface": self._condition_number(surface),
            "dielectric": self._condition_number(dielectric_operator),
        }
        if any(
            not np.isfinite(value) or value > _MAXIMUM_REFERENCE_CONDITION_NUMBER
            for value in conditions.values()
        ):
            raise ValueError(
                "smooth harmonic ddPCM assembly exceeded its condition gate."
            )
        return _DDPCMAssembly(
            weighted_basis=weighted_basis,
            raw_single_layer=raw_single_layer,
            raw_double_layer=raw_double_layer,
            raw_source=raw_source,
            mass=mass,
            surface_operator=surface,
            double_layer=double_layer,
            source_operator=source_operator,
            conductor_limit_operator=conductor_limit,
            dielectric_operator=dielectric_operator,
        )

    @staticmethod
    def _solve_primal(assembly: _DDPCMAssembly, source: Any) -> tuple[Any, ...]:
        boundary_load = assembly.source_operator @ source.reshape(-1)
        vacuum_potential = _torch().linalg.solve(assembly.mass, boundary_load)
        dielectric_rhs = assembly.conductor_limit_operator @ vacuum_potential
        intermediate_potential = _torch().linalg.solve(
            assembly.dielectric_operator, dielectric_rhs
        )
        surface_rhs = assembly.mass @ intermediate_potential
        apparent_charge = _torch().linalg.solve(assembly.surface_operator, surface_rhs)
        return (
            boundary_load,
            vacuum_potential,
            intermediate_potential,
            apparent_charge,
        )

    @staticmethod
    def _solve_adjoint(assembly: _DDPCMAssembly, boundary_load: Any) -> tuple[Any, ...]:
        """Solve the transpose chain for the charging-work scalar.

        The ddPCM primal map is generally nonsymmetric.  These multipliers are
        therefore part of the discrete energy derivative, not an optional
        numerical correction.  They are the dense analogue of the transpose
        solves used by ddX.
        """

        surface_adjoint = _torch().linalg.solve(
            assembly.surface_operator.T, 0.5 * boundary_load
        )
        dielectric_adjoint = _torch().linalg.solve(
            assembly.dielectric_operator.T,
            assembly.mass.T @ surface_adjoint,
        )
        projection_adjoint = _torch().linalg.solve(
            assembly.mass.T,
            assembly.conductor_limit_operator.T @ dielectric_adjoint,
        )
        return surface_adjoint, dielectric_adjoint, projection_adjoint

    @classmethod
    def _response_operators(cls, assembly: _DDPCMAssembly) -> tuple[Any, Any]:
        """Return primal and energy-conjugate source-covector operators.

        Both matrices act on flattened source coefficients and return source
        covectors, i.e. the registered pairing metric has already been
        applied.  The first operator is obtained from the primal apparent
        charge alone.  The second includes the transpose/KKT branch and is the
        Hessian of the discrete scalar exposed by :meth:`drive`.
        """

        source_operator = assembly.source_operator
        vacuum_potential = _torch().linalg.solve(assembly.mass, source_operator)
        intermediate_potential = _torch().linalg.solve(
            assembly.dielectric_operator,
            assembly.conductor_limit_operator @ vacuum_potential,
        )
        apparent_charge = _torch().linalg.solve(
            assembly.surface_operator,
            assembly.mass @ intermediate_potential,
        )
        _, _, projection_adjoint = cls._solve_adjoint(assembly, source_operator)
        primal = -source_operator.T @ apparent_charge
        energy_cotangent = source_operator.T @ (
            -0.5 * apparent_charge - projection_adjoint
        )
        return primal, energy_cotangent

    def _energy_torch(self, positions: Any, source: Any):
        self.configuration_sha256()
        assembly = self._assemble_ddpcm_torch(positions)
        boundary_load, _, _, apparent_charge = self._solve_primal(assembly, source)
        return -0.5 * (boundary_load @ apparent_charge)

    def debug_ddpcm_matrices(self, geometry: object) -> dict[str, np.ndarray]:
        """Return detached finite-dielectric matrices for tests and audits."""

        positions = self._positions_tensor(
            geometry,
            atom_count=len(self._radii_angstrom),
            requires_grad=False,
        )
        assembly = self._assemble_ddpcm_torch(positions)
        return {
            name: np.asarray(value.detach().cpu(), dtype=float).copy()
            for name, value in assembly._asdict().items()
        }

    @staticmethod
    def _residual_record(
        residual: Any, right_hand_side: Any, *, unit: str
    ) -> dict[str, object]:
        absolute = float(_torch().linalg.vector_norm(residual).detach().cpu())
        right_hand_side_norm = float(
            _torch().linalg.vector_norm(right_hand_side).detach().cpu()
        )
        if right_hand_side_norm > np.finfo(float).tiny:
            relative = absolute / right_hand_side_norm
        else:
            relative = 0.0 if absolute == 0.0 else float("inf")
        return {
            "absolute": absolute,
            "right_hand_side_norm": right_hand_side_norm,
            "relative": relative,
            "scaled": absolute / max(right_hand_side_norm, 1.0),
            "unit": unit,
        }

    def stationarity_audit(self, geometry: object, source: object) -> dict[str, object]:
        """Audit primal/KKT stationarity and the energy-cotangent identity."""

        values = self.source_space.validate(
            source,
            atom_count=len(self._radii_angstrom),
            name="ddPCM stationarity audit source",
        )
        positions = self._positions_tensor(
            geometry,
            atom_count=len(self._radii_angstrom),
            requires_grad=False,
        )
        source_tensor = self._source_tensor(values, requires_grad=False)
        assembly = self._assemble_ddpcm_torch(positions)
        boundary, vacuum, intermediate, charge = self._solve_primal(
            assembly, source_tensor
        )

        charge_adjoint, dielectric_adjoint, projection_adjoint = self._solve_adjoint(
            assembly, boundary
        )

        residuals = {
            "vacuum_projection_primal": self._residual_record(
                assembly.mass @ vacuum - boundary,
                boundary,
                unit="eV/e",
            ),
            "dielectric_primal": self._residual_record(
                assembly.dielectric_operator @ intermediate
                - assembly.conductor_limit_operator @ vacuum,
                assembly.conductor_limit_operator @ vacuum,
                unit="eV/e",
            ),
            "single_layer_primal": self._residual_record(
                assembly.surface_operator @ charge - assembly.mass @ intermediate,
                assembly.mass @ intermediate,
                unit="eV/e",
            ),
            "single_layer_adjoint": self._residual_record(
                assembly.surface_operator.T @ charge_adjoint - 0.5 * boundary,
                0.5 * boundary,
                unit="eV/e",
            ),
            "dielectric_adjoint": self._residual_record(
                assembly.dielectric_operator.T @ dielectric_adjoint
                - assembly.mass.T @ charge_adjoint,
                assembly.mass.T @ charge_adjoint,
                unit="e",
            ),
            "vacuum_projection_adjoint": self._residual_record(
                assembly.mass.T @ projection_adjoint
                - assembly.conductor_limit_operator.T @ dielectric_adjoint,
                assembly.conductor_limit_operator.T @ dielectric_adjoint,
                unit="e",
            ),
        }
        conditions = {
            "mass": self._condition_number(assembly.mass),
            "surface": self._condition_number(assembly.surface_operator),
            "dielectric": self._condition_number(assembly.dielectric_operator),
        }
        primal_operator, energy_cotangent_operator = self._response_operators(assembly)
        primal_operator = np.asarray(primal_operator.detach().cpu(), dtype=float)
        energy_cotangent_operator = np.asarray(
            energy_cotangent_operator.detach().cpu(), dtype=float
        )
        source_flat = np.asarray(values, dtype=float).reshape(-1)
        kkt_cotangent = energy_cotangent_operator @ source_flat
        autograd_cotangent = self.pairing.field_to_source_dual(
            self.drive(geometry, values)
        ).reshape(-1)
        direct_primal_cotangent = primal_operator @ source_flat

        component_count = self.source_space.component_count
        monopole_indices = np.arange(0, source_flat.size, component_count)
        primal_monopole_operator = primal_operator[
            np.ix_(monopole_indices, monopole_indices)
        ]
        energy_monopole_operator = energy_cotangent_operator[
            np.ix_(monopole_indices, monopole_indices)
        ]
        charge_tangent_projector = np.eye(len(monopole_indices)) - np.ones(
            (len(monopole_indices), len(monopole_indices))
        ) / len(monopole_indices)

        def relative_norm(values: np.ndarray, *references: np.ndarray) -> float:
            scale = max(
                *(float(np.linalg.norm(reference)) for reference in references),
                np.finfo(float).tiny,
            )
            return float(np.linalg.norm(values)) / scale

        primal_asymmetry = primal_monopole_operator - primal_monopole_operator.T
        energy_asymmetry = energy_monopole_operator - energy_monopole_operator.T
        projected_primal = (
            charge_tangent_projector
            @ primal_monopole_operator
            @ charge_tangent_projector
        )
        projected_energy = (
            charge_tangent_projector
            @ energy_monopole_operator
            @ charge_tangent_projector
        )
        symmetric_primal = 0.5 * (primal_monopole_operator + primal_monopole_operator.T)
        kkt_autograd_error = kkt_cotangent - autograd_cotangent
        primal_cotangent_error = direct_primal_cotangent - kkt_cotangent
        scalar_energy = float((-0.5 * boundary @ charge).detach().cpu())
        half_coupling_energy = 0.5 * float(source_flat @ kkt_cotangent)
        energy_cotangent_gate = (
            relative_norm(
                energy_asymmetry,
                energy_monopole_operator,
                energy_monopole_operator.T,
            )
            <= _COTANGENT_CLOSURE_THRESHOLD
            and relative_norm(
                projected_energy - projected_energy.T,
                projected_energy,
                projected_energy.T,
            )
            <= _COTANGENT_CLOSURE_THRESHOLD
            and relative_norm(
                energy_monopole_operator - symmetric_primal,
                energy_monopole_operator,
                symmetric_primal,
            )
            <= _COTANGENT_CLOSURE_THRESHOLD
            and relative_norm(
                kkt_autograd_error,
                kkt_cotangent,
                autograd_cotangent,
            )
            <= _COTANGENT_CLOSURE_THRESHOLD
            and abs(scalar_energy - half_coupling_energy)
            <= _COTANGENT_CLOSURE_THRESHOLD
            * max(abs(scalar_energy), abs(half_coupling_energy), 1.0)
        )
        response_audit = {
            "pairing_metric_id": self.pairing.scalar_id,
            "operator_representation": (
                "active point-monopole source-to-source-covector block; "
                "registered pairing applied; l=1 rows/columns are identically zero"
            ),
            "primal_response_used_as_provider_field": False,
            "primal_response_is_energy_cotangent": (
                relative_norm(
                    primal_monopole_operator - energy_monopole_operator,
                    primal_monopole_operator,
                    energy_monopole_operator,
                )
                <= _COTANGENT_CLOSURE_THRESHOLD
            ),
            "primal_response_relative_asymmetry": relative_norm(
                primal_asymmetry,
                primal_monopole_operator,
                primal_monopole_operator.T,
            ),
            "primal_charge_tangent_relative_asymmetry": relative_norm(
                projected_primal - projected_primal.T,
                projected_primal,
                projected_primal.T,
            ),
            "energy_cotangent_relative_asymmetry": relative_norm(
                energy_asymmetry,
                energy_monopole_operator,
                energy_monopole_operator.T,
            ),
            "energy_cotangent_charge_tangent_relative_asymmetry": relative_norm(
                projected_energy - projected_energy.T,
                projected_energy,
                projected_energy.T,
            ),
            "energy_cotangent_vs_symmetric_primal_relative_error": relative_norm(
                energy_monopole_operator - symmetric_primal,
                energy_monopole_operator,
                symmetric_primal,
            ),
            "kkt_vs_autograd_absolute_source_covector_norm": float(
                np.linalg.norm(kkt_autograd_error)
            ),
            "kkt_vs_autograd_relative_error": relative_norm(
                kkt_autograd_error,
                kkt_cotangent,
                autograd_cotangent,
            ),
            "primal_vs_energy_cotangent_absolute_source_covector_norm": float(
                np.linalg.norm(primal_cotangent_error)
            ),
            "primal_vs_energy_cotangent_relative_error": relative_norm(
                primal_cotangent_error,
                direct_primal_cotangent,
                kkt_cotangent,
            ),
            "scalar_energy_eV": scalar_energy,
            "half_energy_cotangent_pairing_eV": half_coupling_energy,
            "half_coupling_absolute_error_eV": abs(
                scalar_energy - half_coupling_energy
            ),
            "threshold": _COTANGENT_CLOSURE_THRESHOLD,
            "gate_passed": energy_cotangent_gate,
        }
        gate = (
            response_audit["gate_passed"]
            and all(
                np.isfinite(record["absolute"])
                and np.isfinite(record["relative"])
                and np.isfinite(record["scaled"])
                and record["absolute"] <= _RESIDUAL_THRESHOLD
                and record["relative"] <= _RESIDUAL_THRESHOLD
                and record["scaled"] <= _RESIDUAL_THRESHOLD
                for record in residuals.values()
            )
            and all(
                np.isfinite(value) and value <= _MAXIMUM_REFERENCE_CONDITION_NUMBER
                for value in conditions.values()
            )
        )
        return {
            "state_dimension_per_block": int(assembly.mass.shape[0]),
            "primal_block_count": 3,
            "adjoint_block_count": 3,
            "dielectric": self._dielectric,
            "finite_dielectric_parameterization": True,
            "residuals": residuals,
            "condition_numbers": conditions,
            "response_operator_audit": response_audit,
            "thresholds": {
                "absolute_residual": _RESIDUAL_THRESHOLD,
                "relative_residual": _RESIDUAL_THRESHOLD,
                "max_rhs_or_one_scaled_residual": _RESIDUAL_THRESHOLD,
                "condition_number": _MAXIMUM_REFERENCE_CONDITION_NUMBER,
                "energy_cotangent_closure": _COTANGENT_CLOSURE_THRESHOLD,
            },
            "gate_passed": gate,
            "capability_admitted": False,
        }


__all__ = [
    "DDX_EQUATION_REFERENCE_COMMIT",
    "SMOOTH_POINT_HARMONIC_DDPCM_FUNCTIONAL_CONTRACT_ID",
    "SMOOTH_POINT_HARMONIC_DDPCM_PROVIDER_ID",
    "SmoothPointChargeHarmonicDDPCMFunctionalCandidate",
]

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, TYPE_CHECKING
from collections.abc import Iterable, Mapping

import numpy as np

import ase.calculators.calculator

from ._batch_types import BatchResult
from ._batch_utils import default_calculate_many
from .analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
    AnalyticOperationResult,
    ScalarPointIdentity,
    identity_record,
)
from .solvent_composition import (
    compose_analytic_hessian,
    compose_direct_hvp,
    finalize_energy_forces,
)

if TYPE_CHECKING:
    import torch


_REGISTRY: dict[str, type] = {}


def register_calculator(cls):
    """Register a calculator class under each name in cls.MODEL_NAMES.

    Raises ValueError on duplicate registration to surface accidental name
    collisions instead of silently overwriting.
    """
    for raw_name in cls.MODEL_NAMES:
        name = raw_name.lower()
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Calculator name '{name}' is already registered to {existing.__name__}; "
                f"refusing to overwrite with {cls.__name__}."
            )
        _REGISTRY[name] = cls
    return cls


def get_registered_calculator(name: str) -> type:
    return _REGISTRY[name.lower()]


def import_calculator_plugin(module_path: str) -> None:
    import importlib
    importlib.import_module(module_path)


def load_calculator_plugins_from_env() -> None:
    import os
    raw = os.environ.get('MAPLE_CALCULATOR_PLUGINS', '')
    for entry in (s.strip() for s in raw.split(',') if s.strip()):
        import_calculator_plugin(entry)


_NONE_OPTIONS = {'', 'none', 'null', 'false', '0'}


IMPLICIT_SOLVENT_FORCE_ERROR = (
    "The selected implicit-solvent provider does not supply the requested "
    "energy-consistent derivative."
)
IMPLICIT_SOLVENT_DERIVATIVE_PROPERTIES = {
    "forces",
    "force",
    "stress",
    "stresses",
    "virial",
    "virials",
    "hessian",
}

def reject_implicit_solvent_derivatives(calculator, properties):
    """Fail fast when a selected provider lacks the requested derivative."""
    if not getattr(calculator, "solvent_correction", None):
        return _property_list(properties)
    normalized = _property_list(properties)
    requested = {str(prop).lower() for prop in normalized}
    correction = calculator.solvent_correction
    supported = set(getattr(correction, "supported_properties", {"energy"}))
    unsupported = requested.intersection(IMPLICIT_SOLVENT_DERIVATIVE_PROPERTIES)
    if "forces" in supported:
        unsupported.difference_update({"forces", "force"})
    if "hessian" in supported:
        unsupported.discard("hessian")
    if unsupported:
        raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
    return normalized


def normalize_none_option(value):
    """Normalize user-facing none-like option values to the literal 'none'."""
    if value is None:
        return 'none'
    text = str(value).strip().lower()
    return 'none' if text in _NONE_OPTIONS else text


def validate_implicit_solvent_choice(implicit, solvent):
    """Normalize and validate implicit-solvent selector pair."""
    implicit_norm = normalize_none_option(implicit)
    solvent_norm = normalize_none_option(solvent)
    if implicit_norm == 'gbsa':
        raise ValueError(
            "The legacy implicit='gbsa' + QEqTorch path is removed because it "
            "is not the audited Route 1 fixed-charge composition. Use the "
            "#charge(...) and #solv(method=gb|pb,experimental=true) provider "
            "framework, or implicit='none' to disable solvent."
        )
    return implicit_norm, solvent_norm


def parse_bool_option(value, *, name='option'):
    """Parse a model-option flag into a real bool.

    Accepts actual bools and the strings true/false/1/0/yes/no/on/off
    (case-insensitive). Raises ValueError on anything else so a typo like
    d4=flase fails loudly instead of silently enabling the flag — note that
    bool('false') is True, the exact trap this guards against.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes', 'on'):
        return True
    if text in ('false', '0', 'no', 'off', ''):
        return False
    raise ValueError(
        f"Cannot parse {name}={value!r} as a boolean; use one of true/false/1/0/yes/no."
    )


EV2HARTREE = 1.0 / 27.211386245988


def _convert_energy_force_units(energy, forces, *, source_unit):
    """Convert backend (energy, forces) to Hartree and Hartree/Å.

    Backends declare MODEL_ENERGY_UNIT honestly. Hartree is a no-op; eV
    multiplies through by EV2HARTREE.
    """
    if source_unit == 'hartree':
        return energy, forces
    if source_unit == 'eV':
        energy_ha = energy * EV2HARTREE
        forces_ha = forces * EV2HARTREE if forces is not None else None
        return energy_ha, forces_ha
    raise ValueError(
        f"Unknown source_unit: {source_unit!r}; expected 'eV' or 'hartree'."
    )


def init_implicit_solvent(calc, implicit, solvent, device):
    """Initialize the legacy constructor surface without enabling old GBSA."""
    validate_implicit_solvent_choice(implicit, solvent)
    calc.solvent_correction = None


def atoms_has_pbc(atoms) -> bool:
    """Return True when an Atoms-like object carries any periodic boundary."""
    return atoms is not None and bool(np.any(getattr(atoms, 'pbc', False)))


def reject_periodic_atoms(atoms, backend_name: str) -> None:
    """Fail loudly for molecular wrappers that do not implement PBC graphs."""
    if atoms_has_pbc(atoms):
        raise NotImplementedError(
            f"{backend_name} is a no-PBC molecular wrapper. "
            "Use UMA or a backend-native PBC calculator for periodic systems."
        )


def numerical_hessian_from_atoms(
    calc,
    atoms,
    delta=0.002,
    *,
    return_diagnostics=False,
):
    """Numerical Hessian via central finite difference on forces.

    Polymorphic: works for any calculator with the ASE protocol
    (calc.calculate(atoms, properties=['forces'], system_changes=...) writes
    calc.results['forces'] as a (N, 3) ndarray). Returns float64 ndarray of
    shape (3N, 3N). With ``return_diagnostics=True``, also return absolute and
    scale-aware raw pre-symmetrization diagnostics so validation workflows can
    audit finite-difference quality instead of observing a symmetry imposed by
    construction.
    """
    from ase.constraints import FixAtoms
    from ase.calculators.calculator import all_changes

    old_results = dict(getattr(calc, 'results', {}) or {})
    old_atoms = getattr(calc, 'atoms', None)
    missing = object()
    old_solvation_result = getattr(calc, 'solvation_result', missing)
    try:
        delta = float(delta)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError(
                "Numerical Hessian displacement must be positive and finite."
            )
        N = len(atoms)
        pos0 = atoms.get_positions().copy()
        constraints = list(getattr(atoms, 'constraints', []) or [])
        unsupported_constraints = [
            type(constraint).__name__
            for constraint in constraints
            if not isinstance(constraint, FixAtoms)
        ]
        if unsupported_constraints:
            raise NotImplementedError(
                "Numerical Cartesian Hessians currently support ASE FixAtoms "
                "constraints only. Reduced-coordinate/projection behavior is "
                "not implemented for: "
                + ", ".join(sorted(set(unsupported_constraints)))
                + "."
            )
        fixed = {
            i
            for c in constraints
            if isinstance(c, FixAtoms)
            for i in c.get_indices()
        }
        movable = [i for i in range(N) if i not in fixed]

        H = np.zeros((3 * N, 3 * N), dtype=np.float64)
        if not movable:
            diagnostics = {
                "cartesian_displacement_angstrom": delta,
                "maximum_absolute_raw_hessian_hartree_per_angstrom2": 0.0,
                "maximum_raw_asymmetry_hartree_per_angstrom2": 0.0,
                "raw_hessian_frobenius_norm_hartree_per_angstrom2": 0.0,
                "raw_asymmetry_frobenius_norm_hartree_per_angstrom2": 0.0,
                "relative_raw_asymmetry_frobenius": 0.0,
            }
            return (H, diagnostics) if return_diagnostics else H

        def force_at(positions):
            at = atoms.copy()
            at.set_positions(positions)
            if getattr(atoms, 'constraints', None):
                at.set_constraint(atoms.constraints)
            calc.calculate(at, properties=['forces'], system_changes=all_changes)
            return np.asarray(calc.results['forces'], dtype=np.float64)

        for a in movable:
            for k in range(3):
                # Displacing DOF j and measuring all forces yields -dF_i/dx_j = H[i, j],
                # i.e. column j of the Hessian. Fill the column, then symmetrize to
                # absorb finite-difference noise (the exact Hessian is symmetric).
                col = 3 * a + k
                pos_p = pos0.copy(); pos_p[a, k] += delta
                Fp = force_at(pos_p)
                pos_m = pos0.copy(); pos_m[a, k] -= delta
                Fm = force_at(pos_m)
                H[:, col] = (-(Fp - Fm) / (2.0 * delta)).reshape(-1)

        raw_asymmetry = H - H.T
        maximum_absolute_raw_hessian = float(np.max(np.abs(H)))
        maximum_raw_asymmetry = float(np.max(np.abs(raw_asymmetry)))
        raw_hessian_frobenius_norm = float(np.linalg.norm(H, ord="fro"))
        raw_asymmetry_frobenius_norm = float(
            np.linalg.norm(raw_asymmetry, ord="fro")
        )
        relative_raw_asymmetry_frobenius = (
            raw_asymmetry_frobenius_norm / raw_hessian_frobenius_norm
            if raw_hessian_frobenius_norm
            else 0.0
        )
        H = 0.5 * (H + H.T)
        if fixed:
            # PHVA embedding: a frozen atom contributes no Hessian row/column.
            # Symmetrization would otherwise smear the movable->fixed force
            # couplings (read from the raw, unconstrained results['forces']) into
            # the fixed DOFs; zero them so fixed atoms decouple cleanly. No-op when
            # there are no constraints, so the common path is unchanged.
            fixed_dofs = [3 * i + k for i in sorted(fixed) for k in range(3)]
            H[fixed_dofs, :] = 0.0
            H[:, fixed_dofs] = 0.0
        diagnostics = {
            "cartesian_displacement_angstrom": delta,
            "maximum_absolute_raw_hessian_hartree_per_angstrom2": (
                maximum_absolute_raw_hessian
            ),
            "maximum_raw_asymmetry_hartree_per_angstrom2": (
                maximum_raw_asymmetry
            ),
            "raw_hessian_frobenius_norm_hartree_per_angstrom2": (
                raw_hessian_frobenius_norm
            ),
            "raw_asymmetry_frobenius_norm_hartree_per_angstrom2": (
                raw_asymmetry_frobenius_norm
            ),
            "relative_raw_asymmetry_frobenius": (
                relative_raw_asymmetry_frobenius
            ),
        }
        return (H, diagnostics) if return_diagnostics else H
    finally:
        calc.results = old_results
        calc.atoms = old_atoms
        if old_solvation_result is missing:
            if hasattr(calc, 'solvation_result'):
                del calc.solvation_result
        else:
            calc.solvation_result = old_solvation_result


def hessian_via_double_autograd(energy_fn, leaf):
    """Row-by-row analytic Hessian via double autograd.

    Shared by every CalcABC backend whose ``_analytic_hessian`` builds the
    Hessian one column at a time. ``leaf`` is the position tensor created with
    ``requires_grad=True``; ``energy_fn()`` must return a scalar energy tensor
    that depends on ``leaf`` **already in the target unit** (eV-native backends
    multiply by ``EV2HARTREE`` inside ``energy_fn``; ANI is Hartree-native).

    Returns a ``(3N, 3N)`` float ndarray. Polymorphic over ``leaf`` shape —
    ``(N, 3)`` for MACE-style backends and ``(1, N, 3)`` for ANI both flatten to
    ``3N`` — because it only ever reshapes gradients to 1-D.
    """
    import torch

    energy = energy_fn()
    n = leaf.numel()
    hessian = torch.zeros((n, n), dtype=leaf.dtype, device=leaf.device)
    grad = torch.autograd.grad(energy, leaf, create_graph=True)[0].reshape(-1)
    for i in range(n):
        grad2 = torch.autograd.grad(grad[i], leaf, retain_graph=True)[0].reshape(-1)
        hessian[i, :] = grad2
    return hessian.detach().cpu().numpy()


def _property_list(properties):
    return ['energy'] if properties is None else properties


def calculator_execution(calculator) -> dict[str, Any]:
    """Report component placement without equating MLIP GPU with solver GPU.

    Parameter metadata is inspected without transfers or inference. Keeping
    this live (rather than caching it at construction) reflects a backend's
    numerical-derivative precision preparation. Unknown metadata stays unknown.
    """
    model = getattr(calculator, 'model', None)
    parameter = None
    parameters = getattr(model, 'parameters', None)
    if callable(parameters):
        candidates = parameters()
        if isinstance(candidates, Iterable):
            parameter = next(iter(candidates), None)
    correction = getattr(calculator, 'solvent_correction', None)
    provider = getattr(correction, 'provider', None)
    provenance = getattr(provider, 'provenance', {}) or {}
    if callable(provenance):
        provenance = provenance()
    if not isinstance(provenance, Mapping):
        raise TypeError('Provider execution provenance must be a mapping.')
    wrapper_provenance = provenance
    underlying_provenance = provenance.get('underlying_provider')
    if isinstance(underlying_provenance, Mapping):
        provenance = underlying_provenance
    device = getattr(calculator, 'device', None)
    dtype = getattr(parameter, 'dtype', getattr(calculator, 'dtype', None))
    parameter_device = getattr(parameter, 'device', None)
    result: dict[str, Any] = {
        'model': {
            'backend': type(calculator).__name__,
            'requested_device': getattr(calculator, 'requested_device', None),
            'reported_device': str(device) if device is not None else None,
            'first_parameter_device': str(parameter_device) if parameter_device is not None else None,
            'effective_dtype': str(dtype) if dtype is not None else None,
            'hessian_mode': getattr(calculator, 'hessian', None),
        },
        'solvent': None,
    }
    if provider is not None:
        result['solvent'] = {
            'provider': provenance.get('provider', type(provider).__name__),
            'platform': provenance.get('platform', 'CPU'),
            'platform_properties': provenance.get('platform_properties', {}),
            'selection': {
                **(provenance.get('execution') or {}),
                'testing_only': wrapper_provenance.get('testing_only', False),
                'native_force': wrapper_provenance.get(
                    'native_force', 'forces' in getattr(provider, 'supported_properties', ())
                ),
                'numerical_force': wrapper_provenance.get('numerical_force', False),
            },
        }
    return result


class CalcABC(ase.calculators.calculator.Calculator):
    # Protocol attributes — each subclass overrides what's relevant.
    MODEL_NAMES: tuple = ()
    MODEL_ENERGY_UNIT: str = 'eV'
    SUPPORTED_HESSIAN_MODES: tuple = ('numerical',)
    SUPPORTS_CHARGE_MULT: bool = False
    SUPPORTS_PBC: bool = False
    SUPPORTS_IMPLICIT_SOLVATION: bool = False
    ANALYTIC_IMPLICIT_HESSIAN_MODELS: tuple = ()
    CHECKPOINT_FILENAME: dict | None = None
    REQUIRES_LOCAL_MODEL_FILE: bool = False
    # None keeps legacy/plugins permissive. Shipped backends set an explicit
    # tuple so input typos fail before a model is loaded.
    OPTION_KEYS: tuple | None = None
    # Constructor kwarg that accepts an explicit user model_path, if any.
    MODEL_PATH_OPTION: str | None = None
    # Every CalcABC backend satisfies the return-value batch contract through
    # the safe sequential fallback below. Native model-level batch backends
    # override calculate_many and set this flag to True.
    supports_batch_energy_forces: bool = False
    device: Any
    output: Any
    solvent_correction: Any
    execution_provenance: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.analytic_derivative_admission: AnalyticDerivativeAdmission | None = None
        self.analytic_composition_admission: AnalyticCompositionAdmission | None = None
        self.analytic_implicit_derivatives_admitted = False

    def _reject_unsupported_pbc(self, atoms) -> None:
        if not self.SUPPORTS_PBC:
            reject_periodic_atoms(atoms, type(self).__name__)

    @staticmethod
    def _normalize_properties(properties):
        return _property_list(properties)

    @staticmethod
    def _total_charge_from_atoms(atoms) -> float:
        charge = getattr(atoms, "info", {}).get("charge", None)
        if charge is not None:
            return float(charge)

        if hasattr(atoms, "get_initial_charges"):
            initial_charges = np.asarray(atoms.get_initial_charges(), dtype=float)
            if initial_charges.size and np.all(np.isfinite(initial_charges)):
                return float(initial_charges.sum())

        return 0.0

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=ase.calculators.calculator.all_changes,
    ):
        target_atoms = atoms if atoms is not None else getattr(self, 'atoms', None)
        self._reject_unsupported_pbc(target_atoms)
        properties = reject_implicit_solvent_derivatives(self, properties)
        super().calculate(atoms, properties, system_changes)
        return target_atoms

    def calculate_many(
        self,
        atoms_list,
        properties=("energy", "forces"),
    ) -> BatchResult:
        """Evaluate structures through a uniform, result-driven batch API.

        The base implementation deliberately calls each backend's validated
        single-structure path. Backends may override this only when their native
        model supports an equivalent energy/force batch without changing units,
        ordering, PBC policy, or implicit-solvent semantics.
        """
        return default_calculate_many(
            self,
            atoms_list,
            properties,
        )

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        """Translate input-header options into ctor kwargs. Backends override."""
        return {}

    def admit_analytic_composition(
        self,
        admission: AnalyticCompositionAdmission,
    ) -> None:
        """Bind an installed, exact gas+solvent qualification to this calculator.

        Runtime graph inspection can verify a cell but cannot create scientific
        qualification.  Callers therefore supply an admission selected from an
        installed allowlist; this method only validates and binds it.
        """
        if not isinstance(admission, AnalyticCompositionAdmission):
            raise TypeError("admission must be an AnalyticCompositionAdmission.")
        gas_admission = getattr(self, "analytic_derivative_admission", None)
        correction = getattr(self, "solvent_correction", None)
        solvent_admission = getattr(
            correction, "analytic_derivative_admission", None
        )
        if gas_admission != admission.gas_admission:
            raise AnalyticIdentityError(
                "The calculator's gas admission does not match the composition profile."
            )
        if solvent_admission != admission.solvent_admission:
            raise AnalyticIdentityError(
                "The correction's solvent admission does not match the composition profile."
            )

        # Mutate admission flags only after every immutable component check passes.
        self.analytic_composition_admission = admission
        self.analytic_implicit_derivatives_admitted = True
        if correction is not None:
            correction.analytic_task_derivatives_admitted = True

    @staticmethod
    def _require_component_operation(
        operation: AnalyticOperationResult,
        *,
        expected: str,
    ) -> ScalarPointIdentity:
        if not isinstance(operation, AnalyticOperationResult):
            raise TypeError(f"{expected} must be an AnalyticOperationResult.")
        if operation.operation != expected:
            raise ValueError(
                f"Expected analytic operation {expected!r}, got {operation.operation!r}."
            )
        if not isinstance(operation.identity, ScalarPointIdentity):
            raise AnalyticIdentityError(
                "An already composed analytic result cannot be reused as a gas component."
            )
        if operation.identity.cell.component != "gas":
            raise AnalyticIdentityError("Analytic gas operation has a non-gas identity.")
        return operation.identity

    def _finalize_results(
        self,
        atoms,
        *,
        energy=None,
        forces=None,
        hessian=None,
        unit=None,
        gas_energy_forces: AnalyticOperationResult | None = None,
        gas_hessian: AnalyticOperationResult | None = None,
    ):
        """Single entry: unit conversion + implicit-solvent + write self.results.

        Backends pass the pure model outputs (in the unit declared by
        MODEL_ENERGY_UNIT). This method converts to Hartree, then optionally
        adds the implicit-solvent correction, then writes self.results. Strict
        analytic adapters instead pass operation-stamped Hartree values through
        ``gas_energy_forces`` and, only when requested, ``gas_hessian``.
        """
        gas_point = None
        if gas_energy_forces is not None:
            gas_point = self._require_component_operation(
                gas_energy_forces,
                expected="energy_forces",
            )
        gas_hessian_point = None
        if gas_hessian is not None:
            if hessian is not None or gas_energy_forces is None:
                raise ValueError(
                    "Typed gas Hessians require typed gas energy/forces and cannot "
                    "be mixed with a raw Hessian."
                )
            gas_hessian_point = self._require_component_operation(
                gas_hessian,
                expected="hessian",
            )
            if gas_hessian_point != gas_point:
                raise AnalyticIdentityError(
                    "Gas energy/force and Hessian operations do not share one scalar point."
                )
        precision = getattr(self, 'inference_precision_provenance', None)
        if precision is not None:
            if not isinstance(precision, dict):
                raise TypeError('Inference precision provenance must be a dictionary.')
        execution_provenance = calculator_execution(self)
        gas_admission = (
            getattr(self, "analytic_derivative_admission", None)
            if gas_energy_forces is not None
            else None
        )
        result_payload, structured_solvation_result = finalize_energy_forces(
            atoms,
            correction=getattr(self, "solvent_correction", None),
            energy=energy,
            forces=forces,
            source_unit=(
                None
                if gas_energy_forces is not None
                else unit if unit is not None else self.MODEL_ENERGY_UNIT
            ),
            gas_operation=gas_energy_forces,
            raw_hessian=hessian,
            composition_admission=getattr(
                self, "analytic_composition_admission", None
            ),
            gas_admission=gas_admission,
            execution_provenance=execution_provenance,
            precision_provenance=precision,
        )
        hessian_provenance = None
        if gas_hessian is not None:
            hessian_ha, hessian_provenance = compose_analytic_hessian(
                atoms,
                gas_operation=gas_hessian,
                correction=getattr(self, "solvent_correction", None),
                composition_admission=getattr(
                    self, "analytic_composition_admission", None
                ),
                gas_admission=gas_admission,
            )
            result_payload["hessian"] = hessian_ha

        # Commit only after conversion, all provider calls, and composition
        # have succeeded, so a failed request cannot expose a partial state.
        self.results = result_payload
        self.execution_provenance = execution_provenance
        if structured_solvation_result is None:
            if hasattr(self, "solvation_result"):
                del self.solvation_result
        else:
            self.solvation_result = structured_solvation_result
        if hessian_provenance is not None:
            self.last_analytic_hessian_provenance = hessian_provenance

    def prepare_numerical_derivatives(self) -> float | None:
        """Prepare backend precision before finite differences, with an optional step hint."""
        return None

    def prepare_analytic_derivatives(self) -> None:
        """Prepare backend precision for an admitted analytic composition."""
        raise NotImplementedError(
            f"{type(self).__name__} has no admitted analytic implicit-solvent "
            "derivative preparation."
        )

    def get_hessian(self, atoms, delta: float | None = None):
        """Return the gas or complete composed-potential Hessian.

        Analytic composition is admitted only when both the gas backend and the
        attached solvent correction expose analytic Hessians.  The numerical
        path differentiates the calculator's reported forces, so it includes
        any attached energy-consistent solvent force exactly once.
        """
        self._reject_unsupported_pbc(atoms)
        mode = getattr(self, 'hessian', self.SUPPORTED_HESSIAN_MODES[0])
        if mode == 'analytic':
            correction = getattr(self, 'solvent_correction', None)
            if correction is not None and self.analytic_composition_admission is not None:
                gas_operation = self._analytic_hessian_operation(atoms)
                composed_hessian, provenance = compose_analytic_hessian(
                    atoms,
                    gas_operation=gas_operation,
                    correction=correction,
                    composition_admission=self.analytic_composition_admission,
                    gas_admission=getattr(
                        self, "analytic_derivative_admission", None
                    ),
                )
                self.last_analytic_hessian_provenance = provenance
                self.last_numerical_hessian_diagnostics = None
                return composed_hessian

            gas_hessian = np.asarray(self._analytic_hessian(atoms), dtype=np.float64)
            expected_shape = (3 * len(atoms),) * 2
            if gas_hessian.shape != expected_shape or not np.isfinite(gas_hessian).all():
                raise ValueError(
                    "The analytic gas Hessian must be a finite (3N, 3N) array."
                )
            if correction is None:
                self.last_analytic_hessian_provenance = {
                    "composition": "gas-analytic",
                    "gas_backend": type(self).__name__,
                }
                self.last_numerical_hessian_diagnostics = None
                return gas_hessian
            if self.analytic_implicit_derivatives_admitted is not True:
                raise NotImplementedError(
                    IMPLICIT_SOLVENT_FORCE_ERROR
                    + " The gas backend has not passed analytic implicit-solvent "
                    "derivative preparation."
                )
            solvent_hessian_fn = getattr(correction, "get_hessian", None)
            if not callable(solvent_hessian_fn):
                raise NotImplementedError(
                    IMPLICIT_SOLVENT_FORCE_ERROR
                    + " The attached correction has no admitted analytic solvent Hessian; "
                    "use hessian=numerical for the complete composed-force fallback."
                )
            solvent_hessian = np.asarray(
                solvent_hessian_fn(atoms), dtype=np.float64
            )
            if (
                solvent_hessian.shape != expected_shape
                or not np.isfinite(solvent_hessian).all()
            ):
                raise ValueError(
                    "The analytic solvent Hessian must be a finite (3N, 3N) array."
                )
            solvent_provenance = dict(
                getattr(correction, "last_derivative_provenance", {}) or {}
            )
            solvent_provider = str(
                solvent_provenance.get("provider", "solvent")
            )
            self.last_analytic_hessian_provenance = {
                "composition": f"gas-analytic+{solvent_provider}-analytic",
                "gas_backend": type(self).__name__,
                "solvent": solvent_provenance,
            }
            self.last_numerical_hessian_diagnostics = None
            return gas_hessian + solvent_hessian
        if mode == 'numerical':
            correction = getattr(self, 'solvent_correction', None)
            if correction is not None and "forces" not in set(
                getattr(correction, "supported_properties", {"energy"})
            ):
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            recommended_step = self.prepare_numerical_derivatives()
            resolve_outer_step = getattr(
                correction, "resolve_outer_curvature_step", None
            )
            if correction is not None and callable(resolve_outer_step):
                numerical_solvent = (
                    getattr(correction, "force_mode", None) == "numerical"
                )
                delta = resolve_outer_step(
                    task_delta=None if numerical_solvent else delta,
                    backend_hint=recommended_step,
                    task=getattr(correction, "task", "freq"),
                )
                preflight = getattr(
                    correction, "preflight_cartesian_outer_displacements", None
                )
                if callable(preflight):
                    preflight(atoms, delta)
            elif delta is None:
                delta = 0.002 if recommended_step is None else recommended_step
            reserve_operation = getattr(
                getattr(correction, "provider", None),
                "reserve_force_operation",
                None,
            )
            operation_context = (
                reserve_operation(
                    atom_count=len(atoms),
                    force_call_count=2 * 3 * len(atoms),
                    operation=(
                        "frequency"
                        if getattr(correction, "task", "freq") == "freq"
                        else "prfo-hessian"
                    ),
                )
                if callable(reserve_operation)
                else nullcontext()
            )
            with operation_context:
                hessian, diagnostics = numerical_hessian_from_atoms(
                    self, atoms, delta, return_diagnostics=True
                )
            if not np.isfinite(hessian).all():
                raise ValueError('Numerical Hessian contains non-finite values.')
            committed_diagnostics = dict(diagnostics)
            if correction is not None and hasattr(
                correction, "last_outer_curvature_resolution"
            ):
                committed_diagnostics["outer_step_resolution"] = dict(
                    correction.last_outer_curvature_resolution
                )
            self.last_numerical_hessian_diagnostics = committed_diagnostics
            self.last_analytic_hessian_provenance = None
            return hessian
        raise ValueError(f"Unknown hessian mode: {mode!r}")

    def _analytic_hessian(self, atoms):
        """Backend autograd Hessian. Override in subclasses that can autodiff."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytic Hessian; "
            "set hessian='numerical' or override _analytic_hessian."
        )

    def _analytic_energy_forces_operation(
        self,
        atoms,
        *,
        need_forces: bool,
    ) -> AnalyticOperationResult:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a typed analytic E/F operation."
        )

    def _analytic_hessian_operation(self, atoms) -> AnalyticOperationResult:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a typed analytic Hessian operation."
        )

    def _analytic_hvp_operation(self, atoms, n: np.ndarray) -> AnalyticOperationResult:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a typed direct-HVP operation."
        )


    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")

    def get_hvp(self, atoms, n: np.ndarray):
        """Return direct ``(Hn, forces, energy)`` at one immutable center.

        Backends implement only ``_analytic_hvp_operation`` for their actual
        scalar graph.  This method performs composition and never materializes
        a dense Hessian or catches an AD failure to invoke finite differences.
        """
        self._reject_unsupported_pbc(atoms)
        gas = self._analytic_hvp_operation(atoms, n)
        gas_point = self._require_component_operation(gas, expected="hvp")
        correction = getattr(self, "solvent_correction", None)
        if correction is None:
            result, provenance = compose_direct_hvp(
                atoms,
                n,
                gas_operation=gas,
                gas_admission=getattr(self, "analytic_derivative_admission", None),
            )
            self.last_hvp_provenance = provenance
            return result

        directional_fn = getattr(correction, "get_directional_derivatives", None)
        if not callable(directional_fn):
            raise AnalyticCapabilityError(
                "The attached solvent correction has no direct-HVP entry point."
            )

        if self.analytic_composition_admission is not None:
            result, provenance = compose_direct_hvp(
                atoms,
                n,
                gas_operation=gas,
                correction=correction,
                composition_admission=self.analytic_composition_admission,
                gas_admission=getattr(self, "analytic_derivative_admission", None),
            )
            self.last_hvp_provenance = provenance
            return result

        # Preserve the previously admitted OpenMM-Reference/Torch-derivative
        # compatibility route without presenting it as strict same-runtime
        # composition.  New independent providers never enter this branch.
        if (
            self.analytic_implicit_derivatives_admitted is not True
            or getattr(correction, "analytic_task_derivatives_admitted", False)
            is not True
        ):
            raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
        solvent = directional_fn(atoms, np.asarray(n, dtype=np.float64))
        hvp = np.asarray(gas.hvp_hartree_per_angstrom2) + np.asarray(
            solvent.hvp_hartree_per_angstrom2
        )
        forces = np.asarray(gas.forces_hartree_per_angstrom) + np.asarray(
            solvent.forces_hartree_per_angstrom
        )
        energy = float(gas.energy_hartree) + float(solvent.energy_hartree)
        if (
            hvp.shape != (3 * len(atoms),)
            or forces.shape != (len(atoms), 3)
            or not np.isfinite(hvp).all()
            or not np.isfinite(forces).all()
            or not np.isfinite(energy)
        ):
            raise ValueError("Legacy split-runtime HVP returned invalid values.")
        self.last_hvp_provenance = {
            "composition": "legacy-split-runtime-compatibility",
            "gas": identity_record(gas_point),
            "solvent": dict(getattr(solvent, "provenance", {}) or {}),
            "strict_same_runtime_scalar": False,
        }
        return hvp, forces.reshape(-1), energy

    def implicit_solv_init(self, implicit: str, solvent: str):
        init_implicit_solvent(self, implicit, solvent, self.device)
    
    def implicit_solv_energy(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Compute implicit solvent correction energy if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Implicit solvent correction energy in Hartree.
        """
        if not hasattr(self.solvent_correction, 'evaluate'):
            raise RuntimeError(
                "Unstructured legacy implicit-solvent corrections are no "
                "longer supported; attach an ImplicitSolvationCorrection."
            )
        return self.solvent_correction.evaluate(atoms).energy_hartree

    def implicit_solv_energy_and_force(self, atoms: ase.Atoms) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute implicit solvent correction energy and forces if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Implicit solvent correction energy in Hartree and forces in Hartree/Å.
        """
        if hasattr(self.solvent_correction, 'evaluate'):
            result = self.solvent_correction.evaluate(atoms, need_forces=True)
            if result.forces_hartree_per_angstrom is None:
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            return result.energy_hartree, result.forces_hartree_per_angstrom
        raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)

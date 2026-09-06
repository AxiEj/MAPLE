"""ASE bridge for the registered pure frozen-source MACE-POLAR ddPCM scalar."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.extra_correction.implicit.route2_domain import (
    validate_route2_domain,
)
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
    Route2SMDProfileSpec,
    route2_smd_profile_spec,
)
from maple.function.route2_solvents import normalize_route2_solvent_name
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
)
from maple.solvation.derivatives.scalar_finite_difference import (
    BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1,
    RichardsonScalarHessian,
)
from maple.solvation.experimental.mace_polar_frozen_ddx import (
    build_smd_mace_polar_frozen_point_ddx_pes,
)
from maple.solvation.models.base import model_input_sha256
from maple.solvation.models.mace_polar import (
    build_official_mace_polar_1_m_radial_gto_adapter,
)

_TEST_PES_FACTORY_TOKEN = object()


class PureMACEPolarDDXCalculator(Calculator):
    """Expose one source-bound total PES through ASE-standard eV units.

    Production construction always loads the official MACE-POLAR-1-M
    CPU/float64 adapter and then calls the exact registered point-l1 ddPCM +
    PySCF SMD-CDS builder.  Fake PES injection is available only through the
    private classmethod used by engineering tests.
    """

    implemented_properties = ["energy", "free_energy", "forces"]
    workflow_kind = "pure-frozen-total-pes"
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTS_PBC = False
    SUPPORTS_CHARGE_MULT = False
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    practical_hessian_step_angstrom = 4.0e-3

    def __init__(
        self,
        *,
        atoms,
        solvent: str,
        device: object = "cpu",
        model: str = "macepolm",
        profile_spec: Route2SMDProfileSpec | None = None,
        _testing_pes: object | None = None,
        _testing_token: object | None = None,
    ) -> None:
        super().__init__()
        normalized_device = getattr(device, "type", device)
        if str(normalized_device).strip().lower() != "cpu":
            raise ValueError("The pure frozen MACE-POLAR workflow requires device=cpu.")
        if str(model).strip().lower() != "macepolm":
            raise ValueError(
                "The pure frozen MACE-POLAR workflow requires model=macepolm."
            )
        spec = profile_spec or route2_smd_profile_spec(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
        )
        if (
            not isinstance(spec, Route2SMDProfileSpec)
            or spec.name != PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
            or spec.execution_route != self.workflow_kind
            or spec.scalar_contract_id
            != EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
            or spec.allowed_response_modes != ("frozen",)
        ):
            raise ValueError("profile_spec is not the registered pure frozen workflow.")
        normalized_solvent = normalize_route2_solvent_name(solvent)
        if not spec.supports_solvent(normalized_solvent):
            raise ValueError(
                f"Route 2 profile={spec.name} does not support solvent={solvent}."
            )
        validate_route2_domain(atoms)

        if _testing_pes is not None and _testing_token is not _TEST_PES_FACTORY_TOKEN:
            raise ValueError("test PES injection is not a public construction path.")
        if _testing_pes is None:
            model_adapter = build_official_mace_polar_1_m_radial_gto_adapter(
                device="cpu"
            )
            hessian_backend = RichardsonScalarHessian(
                coarse_step_angstrom=self.practical_hessian_step_angstrom,
                maximum_error_eV_per_A2=5.0e-2,
                maximum_antisymmetry_eV_per_A2=5.0e-2,
                maximum_topology_step_reductions=0,
                topology_guard_policy=BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1,
                maximum_energy_force_discrepancy_eV_per_A=3.0e-3,
            )
            pes = build_smd_mace_polar_frozen_point_ddx_pes(
                model_adapter,
                tuple(atoms.get_chemical_symbols()),
                solvent=normalized_solvent,
                lmax=15,
                n_lebedev=1202,
                solver_tolerance=1.0e-12,
                eta=0.1,
                n_proc=1,
                hessian_backend=hessian_backend,
            )
        else:
            pes = _testing_pes
        if getattr(pes, "scalar_contract_id", None) != spec.scalar_contract_id:
            raise ValueError(
                "total PES does not implement the profile scalar contract."
            )

        self._device = "cpu"
        self._model_name = "macepolm"
        self._solvent = normalized_solvent
        self._profile_spec = spec
        self._pes = pes
        self.solvent_correction = None
        self.chargecalc = None
        self._configured_symbols = tuple(atoms.get_chemical_symbols())
        self._force_cache_key: tuple[str, str] | None = None
        self._force_evaluation: object | None = None
        self._last_energy_state: object | None = None
        self._hessian_cache_key: tuple[str, str] | None = None
        self._hessian_evaluation: object | None = None

    @classmethod
    def _from_test_pes(
        cls,
        *,
        atoms,
        solvent: str,
        pes: object,
        device: object = "cpu",
        model: str = "macepolm",
        profile_spec: Route2SMDProfileSpec | None = None,
    ) -> "PureMACEPolarDDXCalculator":
        """Build an engineering-test bridge without loading the real model."""

        return cls(
            atoms=atoms,
            solvent=solvent,
            device=device,
            model=model,
            profile_spec=profile_spec,
            _testing_pes=pes,
            _testing_token=_TEST_PES_FACTORY_TOKEN,
        )

    @property
    def last_hessian_evaluation(self) -> object | None:
        return self._hessian_evaluation

    @property
    def device(self) -> str:
        return self._device

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def solvent(self) -> str:
        return self._solvent

    @property
    def profile_spec(self) -> Route2SMDProfileSpec:
        return self._profile_spec

    @property
    def pes(self) -> object:
        return self._pes

    @property
    def last_energy_state(self) -> object | None:
        return self._last_energy_state

    def _validate_atoms(self, atoms) -> tuple[str, str]:
        if (
            self._device != "cpu"
            or self._model_name != "macepolm"
            or self._profile_spec.execution_route != self.workflow_kind
            or self._profile_spec.scalar_contract_id
            != EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
            or self._profile_spec.allowed_response_modes != ("frozen",)
            or not self._profile_spec.supports_solvent(self._solvent)
            or getattr(self._pes, "scalar_contract_id", None)
            != self._profile_spec.scalar_contract_id
        ):
            raise RuntimeError("pure frozen workflow configuration drifted.")
        validate_route2_domain(atoms)
        if tuple(atoms.get_chemical_symbols()) != self._configured_symbols:
            raise ValueError("atoms differ from the calculator's configured symbols.")
        model = getattr(self._pes, "model", None)
        domain = getattr(model, "domain", None)
        validate_atoms = getattr(domain, "validate_atoms", None)
        if callable(validate_atoms):
            validate_atoms(atoms)
        configuration = getattr(self._pes, "configuration_sha256", None)
        configuration_sha256 = (
            str(configuration())
            if callable(configuration)
            else str(configuration or "")
        )
        if not configuration_sha256:
            configuration_sha256 = str(
                getattr(self._pes, "provider_id", type(self._pes).__name__)
            )
        # model_input_sha256 includes geometry plus charge/multiplicity aliases.
        return model_input_sha256(atoms), configuration_sha256

    @staticmethod
    def _reject_hessian_constraints(atoms) -> None:
        if tuple(getattr(atoms, "constraints", ()) or ()):
            raise ValueError(
                "The pure frozen workflow does not support constrained full Hessians."
            )

    def get_property(self, name, atoms=None, allow_calculation=True):
        target = atoms if atoms is not None else self.atoms
        if target is not None:
            self._validate_atoms(target)
        return super().get_property(name, atoms, allow_calculation)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        target = atoms if atoms is not None else self.atoms
        if target is None:
            raise ValueError("PureMACEPolarDDXCalculator requires atoms.")
        cache_key = self._validate_atoms(target)
        super().calculate(target, properties, system_changes)
        if cache_key != self._force_cache_key or self._force_evaluation is None:
            self._last_energy_state = None
            evaluation = self._pes.evaluate_forces(target)
            self._force_cache_key = cache_key
            self._force_evaluation = evaluation
            self._hessian_cache_key = None
            self._hessian_evaluation = None
        evaluation = self._force_evaluation
        state = getattr(evaluation, "central_state", None)
        if state is None:
            # A fake/test PES may expose force and energy evaluations separately.
            state = self._pes.solve(target)
        self._last_energy_state = state
        energy = float(getattr(state, "total_energy_eV"))
        forces = np.asarray(getattr(evaluation, "total_forces_eV_per_A"), dtype=float)
        if forces.shape != (len(target), 3) or not np.all(np.isfinite(forces)):
            raise RuntimeError("pure frozen total-PES forces are invalid.")
        if not np.isfinite(energy):
            raise RuntimeError("pure frozen total-PES energy is invalid.")
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": np.array(forces, copy=True),
        }

    def get_hessian_evaluation(self, atoms):
        self._reject_hessian_constraints(atoms)
        cache_key = self._validate_atoms(atoms)
        if (
            cache_key == self._hessian_cache_key
            and self._hessian_evaluation is not None
        ):
            return self._hessian_evaluation
        previous_results = dict(self.results)
        previous_atoms = self.atoms
        self._hessian_cache_key = None
        self._hessian_evaluation = None
        try:
            evaluation = self._pes.evaluate_hessian(atoms)
        except Exception:
            self.results = previous_results
            self.atoms = previous_atoms
            raise
        self._hessian_cache_key = cache_key
        self._hessian_evaluation = evaluation
        return evaluation

    def get_hessian(self, atoms, delta: float | None = None) -> np.ndarray:
        if delta is not None and float(delta) != self.practical_hessian_step_angstrom:
            raise ValueError(
                "The pure frozen workflow fixes the Richardson coarse step at 0.004 A."
            )
        evaluation = self.get_hessian_evaluation(atoms)
        return np.array(evaluation.hessian_eV_per_A2, dtype=float, copy=True)

    def get_hvp(self, atoms, n: np.ndarray):
        self._reject_hessian_constraints(atoms)
        self._validate_atoms(atoms)
        direction = np.asarray(n, dtype=float)
        expected = (len(atoms), 3)
        if direction.size != 3 * len(atoms) or not np.all(np.isfinite(direction)):
            raise ValueError("HVP direction must contain 3N finite components.")
        evaluation = self._pes.hessian_vector_product(
            atoms, direction.reshape(expected)
        )
        force_evaluation = getattr(evaluation, "central_sample", None)
        forces = getattr(force_evaluation, "forces_eV_per_A", None)
        energy_sample = getattr(force_evaluation, "energy_sample", None)
        energy = getattr(energy_sample, "energy_eV", None)
        if forces is None or energy is None:
            self.calculate(atoms, properties=["energy", "forces"], system_changes=[])
            forces = self.results["forces"]
            energy = self.results["energy"]
        try:
            import torch

            dtype = torch.float64
            hvp = torch.as_tensor(
                np.asarray(evaluation.hvp_eV_per_A2).reshape(-1), dtype=dtype
            )
            force_tensor = torch.as_tensor(np.asarray(forces).reshape(-1), dtype=dtype)
            energy_tensor = torch.as_tensor(float(energy), dtype=dtype)
            return hvp, force_tensor, energy_tensor
        except ImportError:
            return (
                np.asarray(evaluation.hvp_eV_per_A2).reshape(-1),
                np.asarray(forces).reshape(-1),
                float(energy),
            )


def is_pure_mace_polar_workflow_calculator(calculator: object) -> bool:
    """Return whether ``calculator`` is the exact registered workflow facade.

    A user-set ``workflow_kind`` marker is not authority.  The only permitted
    wrapper is MAPLE's actual legacy-unit view, and the underlying calculator,
    canonical profile object, execution route, and scalar binding must all
    match the registry.
    """

    from maple.function.dispatcher.legacy_units import LegacyHartreeJobView

    candidate = calculator
    if type(candidate) is LegacyHartreeJobView:
        candidate = candidate.raw_calculator
    if type(candidate) is not PureMACEPolarDDXCalculator:
        return False
    canonical = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
    )
    return bool(
        candidate.profile_spec is canonical
        and candidate.workflow_kind == canonical.execution_route
        and canonical.execution_route == "pure-frozen-total-pes"
        and canonical.scalar_contract_id
        == EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
        and candidate.device == "cpu"
        and candidate.model == "macepolm"
        and canonical.supports_solvent(candidate.solvent)
        and getattr(candidate.pes, "scalar_contract_id", None)
        == canonical.scalar_contract_id
    )


__all__ = [
    "PureMACEPolarDDXCalculator",
    "is_pure_mace_polar_workflow_calculator",
]

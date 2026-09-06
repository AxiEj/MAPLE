"""Experimental MAPLE/ASE surface for the named-solvent hybrid ddX scalar."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from ase.calculators.calculator import all_changes

from ..calculator_base import (
    EV2HARTREE,
    CalcABC,
    _public_hessian,
    register_calculator,
)
from ._mace_mdp_polar_hybrid_calculator import MACE_MDPPOLARHybridCalculator


@register_calculator
class MACE_MDPPOLARHybridDDXCalculator(CalcABC):
    """Experimental named-solvent ddX/SMD hybrid daily calculator.

    This class is deliberately a separate model identity.  It must not inherit
    the harmonic profile's admission, and it must not be confused with the
    coordinate-incomplete canonical-ADT accuracy candidate.
    """

    implemented_properties = ["energy", "forces", "free_energy"]

    MODEL_NAMES = ("macemdppolarhybridddx",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    FREQUENCY_THERMOCHEMISTRY = "none"
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = (
        "mdp_checkpoint_path",
        "polar_checkpoint_path",
        "solvent",
        "d4",
    )
    MAPLE_DAILY_JOB_CAPABILITIES = (
        "sp",
        "opt:first-order",
        "freq:vibrational-only-experimental",
        "ts:neb-cineb-experimental",
        "hessian:direct-experimental-richardson",
        "virial:direct-nonperiodic-molecular",
    )

    def validate_maple_job(self, *, jobtype: object, params: object = None) -> None:
        """Open only workflows supported by the exposed scalar derivatives."""

        normalized_job = str(jobtype).strip().lower()
        options = params if isinstance(params, dict) else {}
        if normalized_job == "sp":
            return
        if normalized_job == "freq":
            # The frequency driver enforces FREQUENCY_THERMOCHEMISTRY, including
            # direct programmatic calls. Hessian numerical guards still apply.
            return
        if normalized_job == "opt":
            method = str(options.get("method") or "lbfgs").strip().lower()
            if method in {"lbfgs", "sd", "sdcg", "cg"}:
                return
            raise NotImplementedError(
                "macemdppolarhybridddx OPT currently supports only first-order "
                "LBFGS/SD/SDCG/CG methods."
            )
        if normalized_job == "ts":
            method = str(options.get("method") or "").strip().lower()
            refine = options.get("refine")
            if method == "neb" and refine in (None, "cineb"):
                return
            raise NotImplementedError(
                "macemdppolarhybridddx TS supports experimental NEB/CINEB only "
                "(method=neb, optionally refine=cineb). PRFO, NEBTS, Dimer and "
                "other TS methods require separate derivative validation."
            )
        raise NotImplementedError(
            "macemdppolarhybridddx currently supports MAPLE SP and first-order "
            "OPT, vibrational-only FREQ and experimental NEB/CINEB, plus direct "
            "get_hessian()/get_molecular_virial() calls. Solution-phase "
            "thermochemistry, IRC/MD and scan remain closed."
        )

    @classmethod
    def build_kwargs_from_options(
        cls, model: str, model_options: dict[str, Any], *, resolved_model_path=None
    ) -> dict[str, Any]:
        del model, resolved_model_path
        result: dict[str, Any] = {}
        option_keys = cls.OPTION_KEYS
        if option_keys is None:
            raise RuntimeError("hybrid ddX calculator option contract is missing.")
        for name in option_keys:
            value = model_options.get(name)
            if value is not None:
                result["route2_solvent" if name == "solvent" else name] = value
        return result

    _load_hybrid = staticmethod(MACE_MDPPOLARHybridCalculator._load_hybrid)

    def __init__(
        self,
        device: object,
        model: str = "macemdppolarhybridddx",
        implicit: str = "none",
        solvent: str = "water",
        route2_solvent: str | None = None,
        mdp_checkpoint_path: str | Path | None = None,
        polar_checkpoint_path: str | Path | None = None,
        hessian: str = "numerical",
        d4: bool = False,
    ) -> None:
        super().__init__()
        requested_device = str(device).strip().lower()
        if requested_device in {"cuda", "cuda:0", "gpu", "gpu0"}:
            normalized_device = "cuda"
        else:
            raise ValueError(
                "The hybrid ddX daily profile requires device='cuda' for "
                "MACE-POLAR; the frozen MACE-MDP adapter remains CPU/float64."
            )
        if str(model).strip().lower() not in self.MODEL_NAMES:
            raise ValueError("Unsupported hybrid ddX calculator model identity.")
        if str(implicit).strip().lower() not in {"", "none"}:
            raise ValueError(
                "macemdppolarhybridddx already owns ddX and its solvent term; "
                "set implicit='none'."
            )
        if d4:
            raise ValueError(
                "macemdppolarhybridddx already defines its complete experimental "
                "solvent ledger; D4 composition is not admitted."
            )
        normalized_hessian = str(hessian).strip().lower()
        if normalized_hessian not in self.SUPPORTED_HESSIAN_MODES:
            raise ValueError("Hybrid ddX supports only hessian='numerical'.")

        from maple.function.route2_solvents import route2_solvent_spec

        requested_solvent = str(route2_solvent or solvent).strip() or "water"
        if requested_solvent.lower() == "none":
            requested_solvent = "water"
        specification = route2_solvent_spec(requested_solvent)

        mdp_path = (
            Path(mdp_checkpoint_path or Path.home() / ".cache/mace/MACE-MDP.model")
            .expanduser()
            .resolve(strict=True)
        )
        polar_path = (
            Path(polar_checkpoint_path or Path.home() / ".cache/mace/MACEPOLAR1Mmodel")
            .expanduser()
            .resolve(strict=True)
        )
        hybrid, radial = self._load_hybrid(
            device=normalized_device,
            mdp_checkpoint_path=mdp_path,
            polar_checkpoint_path=polar_path,
        )

        self.device = normalized_device
        self.model_name = self.MODEL_NAMES[0]
        self.solvent = specification.name
        self.hessian = normalized_hessian
        self.solvent_correction = None
        self._hybrid = hybrid
        self._radial = radial
        self._pes = None
        self._symbols: tuple[str, ...] | None = None
        self.route2_result = None
        self.hessian_diagnostics = None

    def _build_pes(self, atoms: Any):
        from maple.solvation.derivatives import (
            OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
            RichardsonScalarHessian,
        )
        from maple.solvation.experimental import (
            build_smd_mace_mdp_polar_hybrid_ddx_pes,
        )

        symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
        return build_smd_mace_mdp_polar_hybrid_ddx_pes(
            cast(Any, self._hybrid),
            symbols,
            solvent=self.solvent,
            hessian_backend=RichardsonScalarHessian(
                topology_guard_policy=OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
                maximum_topology_step_reductions=6,
            ),
        )

    def _pes_for(self, atoms: Any):
        self._validate_molecular_state(atoms)
        symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
        if self._pes is None or self._symbols != symbols:
            self._pes = self._build_pes(atoms)
            self._symbols = symbols
        return self._pes

    @staticmethod
    def _validate_molecular_state(atoms: Any) -> None:
        """Do not silently reinterpret charged/spin inputs as this neutral PES."""

        if np.any(atoms.get_pbc()):
            raise ValueError("Hybrid ddX requires a nonperiodic molecule.")
        try:
            charge = float(atoms.info.get("charge", 0))
            mult = float(atoms.info.get("mult", 1))
            multiplicity = float(atoms.info.get("multiplicity", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Hybrid ddX requires numeric neutral singlet metadata.") from exc
        if charge != 0.0 or mult != 1.0 or multiplicity != 1.0:
            raise ValueError("Hybrid ddX supports neutral singlet states only.")
        charges = np.asarray(atoms.get_initial_charges(), dtype=float)
        moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
        if (
            not np.all(np.isfinite(charges))
            or abs(float(charges.sum())) > 1.0e-8
            or not np.all(np.isfinite(moments))
            or np.any(moments != 0.0)
            or int(np.sum(atoms.numbers)) % 2 != 0
        ):
            raise ValueError("Hybrid ddX supports neutral singlet states only.")

    def get_property(self, name, atoms=None, allow_calculation=True):
        # ASE's geometry cache does not track atoms.info charge/multiplicity.
        target = atoms if atoms is not None else self.atoms
        if target is not None:
            self._validate_molecular_state(target)
        return super().get_property(name, atoms, allow_calculation)

    def maple_neb_pes_identity(self, atoms: Any) -> tuple[str, str, str]:
        """Bind all path images to one checkpoint/source/solvent scalar."""

        pes = self._pes_for(atoms)
        return self.model_name, pes.daily_profile_id, pes.configuration_sha256()

    @staticmethod
    def _coordinate_derivative_available(pes: object) -> bool:
        declared = getattr(pes, "coordinate_derivative_available", None)
        if type(declared) is not bool:
            raise RuntimeError(
                "hybrid ddX PES must declare a boolean complete-coordinate-"
                "derivative capability."
            )
        return declared

    @classmethod
    def _require_coordinate_derivative(cls, pes: object, *, property_name: str) -> None:
        if not cls._coordinate_derivative_available(pes):
            raise NotImplementedError(
                f"hybrid ddX {property_name} requires the complete coordinate "
                "derivative, which this configured electrostatic provider does "
                "not expose."
            )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        requested = self._normalize_properties(properties)
        target = super().calculate(atoms, requested, system_changes)
        unsupported = sorted(
            set(str(value).lower() for value in requested)
            - {"energy", "free_energy", "forces"}
        )
        if unsupported:
            raise NotImplementedError(
                "The hybrid ddX ASE result surface exposes E/F; use get_hessian() "
                "for H and get_molecular_virial() for the nonperiodic molecular "
                f"virial. Unsupported properties: {', '.join(unsupported)}."
            )

        pes = self._pes_for(target)
        state = pes.solve(target)
        evaluation = None
        forces = None
        virial = None
        coordinate_derivative_available = self._coordinate_derivative_available(pes)
        if "forces" in requested:
            self._require_coordinate_derivative(pes, property_name="forces")
            evaluation = pes.evaluate_forces(target, central_state=state)
            forces = np.asarray(evaluation.total_forces_eV_per_A, dtype=float)
            virial = pes.molecular_virial(target, force_evaluation=evaluation)

        self._finalize_results(
            target,
            energy=state.total_energy_eV,
            forces=forces,
            unit="eV",
        )
        self.route2_result = evaluation if evaluation is not None else state
        route2 = {
            "profile_id": pes.daily_profile_id,
            "provider_id": pes.provider_id,
            "scalar_contract_id": pes.scalar_contract_id,
            "configuration_sha256": pes.configuration_sha256(),
            "state_sha256": state.state_sha256,
            "electrostatic_root_sha256": state.electrostatic_state.root_sha256,
            "solvent": self.solvent,
            "topology_observation_coverage": state.topology_observation_coverage,
            "unobservable_topology_components": (
                state.unobservable_topology_components
            ),
            "energy_components_eV": {
                "vacuum": state.vacuum_energy_eV,
                "ddx_polarization": state.polarization_energy_eV,
                "smd_cds": state.cds_energy_eV,
                "solvation_total": state.solvation_energy_eV,
            },
            "scientific_status": pes.scientific_status,
            "daily_job_capabilities": self.MAPLE_DAILY_JOB_CAPABILITIES,
            "daily_property_availability": {
                "energy": "experimental",
                "forces": (
                    "experimental-same-scalar-analytic"
                    if coordinate_derivative_available
                    else False
                ),
                "hessian": (
                    "experimental-richardson"
                    if coordinate_derivative_available
                    else False
                ),
                "molecular_virial": (
                    "experimental-nonperiodic"
                    if coordinate_derivative_available
                    else False
                ),
                "periodic_stress": False,
                "strict_variational_functional": False,
                "molecular_dynamics": False,
            },
            "warnings": (
                "experimental daily surface; development MAE and tail gates are "
                "not production-admitted",
                "H/FREQ uses an observed-components-only topology policy for the "
                "PySCF SMD-CDS term",
                "FREQ reports vibrations only, not gas-phase or solution-phase "
                "thermochemistry; NEB/CINEB returns a TS candidate, not a "
                "certified first-order saddle",
            ),
        }
        if evaluation is not None:
            route2.update(
                {
                    "force_evaluation_sha256": evaluation.evaluation_sha256,
                    "force_derivative_kind": (
                        "analytic-block-adjoint-plus-solvent-gradient-v1"
                    ),
                    "adjoint_residual_eV": (
                        evaluation.electrostatic_evaluation.adjoint_residual_ev
                    ),
                }
            )
        if virial is not None:
            route2["molecular_virial_eV"] = np.asarray(
                virial.raw_virial_eV, dtype=float
            )
            route2["molecular_virial_origin_A"] = np.asarray(
                virial.origin_angstrom, dtype=float
            )
            route2["molecular_virial_evaluation_sha256"] = virial.evaluation_sha256
        self.results["route2"] = route2
        self.results["solvation"] = {
            "gas_energy_hartree": state.vacuum_energy_eV * EV2HARTREE,
            "delta_g_solv_hartree": state.solvation_energy_eV * EV2HARTREE,
            "combined_energy_hartree": state.total_energy_eV * EV2HARTREE,
            "provenance": {
                "profile_id": pes.daily_profile_id,
                "solvent": self.solvent,
                "standard_state": (
                    "profile-defined electronic-plus-continuum ledger; no "
                    "thermochemical Gibbs correction"
                ),
            },
        }

    def get_molecular_virial(self, atoms, *, symmetric: bool = False) -> np.ndarray:
        """Return the declared-origin molecular virial in eV.

        This is not periodic ASE stress.  The default origin is the arithmetic
        mean of the molecular coordinates, matching the underlying record.
        """

        pes = self._pes_for(atoms)
        self._require_coordinate_derivative(pes, property_name="molecular virial")
        evaluation = pes.evaluate_forces(atoms)
        virial = pes.molecular_virial(atoms, force_evaluation=evaluation)
        values = virial.symmetric_virial_eV if symmetric else virial.raw_virial_eV
        return np.array(values, dtype=float, copy=True)

    def get_hessian(self, atoms, delta: float = 0.002):
        """Return the total same-scalar Richardson Hessian in eV/angstrom^2."""

        del delta
        self.hessian_diagnostics = None
        pes = self._pes_for(atoms)
        self._require_coordinate_derivative(pes, property_name="Hessian")
        evaluated = pes.evaluate_hessian(atoms)
        values = np.asarray(evaluated.hessian_eV_per_A2, dtype=float)
        expected = (3 * len(atoms), 3 * len(atoms))
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise RuntimeError(
                "Hybrid ddX Hessian is nonfinite or has the wrong shape."
            )
        self.hessian_diagnostics = {
            name: getattr(evaluated, name)
            for name in (
                "evaluation_sha256", "coarse_step_angstrom", "fine_step_angstrom",
                "maximum_error_estimate_eV_per_A2", "maximum_antisymmetry_eV_per_A2",
                "topology_step_reductions_used", "topology_guard_status",
                "topology_observation_coverage", "unobservable_topology_components",
            )
        }
        return _public_hessian(values, source_unit="eV")

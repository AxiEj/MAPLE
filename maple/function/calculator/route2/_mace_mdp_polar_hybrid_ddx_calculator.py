"""Experimental MAPLE/ASE surface for the named-solvent hybrid ddX scalar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator


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
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = (
        "mdp_checkpoint_path",
        "polar_checkpoint_path",
        "route2_solvent",
        "d4",
    )
    MAPLE_DAILY_JOB_CAPABILITIES = (
        "sp",
        "opt:first-order",
        "hessian:direct-experimental-richardson",
        "virial:direct-nonperiodic-molecular",
    )

    def validate_maple_job(self, *, jobtype: object, params: object = None) -> None:
        """Open only workflows supported by the exposed scalar derivatives."""

        normalized_job = str(jobtype).strip().lower()
        options = params if isinstance(params, dict) else {}
        if normalized_job == "sp":
            return
        if normalized_job == "opt":
            method = str(options.get("method") or "lbfgs").strip().lower()
            if method in {"lbfgs", "sd", "sdcg", "cg"}:
                return
            raise NotImplementedError(
                "macemdppolarhybridddx OPT currently supports only first-order "
                "LBFGS/SD/SDCG/CG methods."
            )
        raise NotImplementedError(
            "macemdppolarhybridddx currently supports MAPLE SP and first-order "
            "OPT plus direct get_hessian()/get_molecular_virial() calls. The "
            "existing MAPLE FREQ driver adds gas-phase thermochemistry and is not "
            "a chemically valid solution-phase workflow; FREQ/TS/IRC/MD and scan "
            "remain closed."
        )

    @classmethod
    def build_kwargs_from_options(
        cls, model: str, model_options: dict[str, Any], *, resolved_model_path=None
    ) -> dict[str, Any]:
        del model, resolved_model_path
        result: dict[str, Any] = {}
        for name in cls.OPTION_KEYS:
            value = model_options.get(name)
            if value is not None:
                target = "daily_solvent" if name == "route2_solvent" else name
                result[target] = value
        return result

    @staticmethod
    def _load_hybrid(
        *, device: str, mdp_checkpoint_path: Path, polar_checkpoint_path: Path
    ) -> tuple[object, object]:
        """Build the two frozen checkpoint adapters with audited devices."""

        from maple.solvation.models import (
            MACEPolarOriginalSourceNativeFieldAdapter,
            build_mace_mdp_anchored_mace_polar_hybrid,
            build_mace_mdp_moment_adapter,
            build_official_mace_polar_1_m_radial_gto_adapter,
        )
        from maple.solvation.models.runtime.analytic_gaussian_multipole import (
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
        )

        mdp = build_mace_mdp_moment_adapter(
            checkpoint_path=mdp_checkpoint_path,
            device="cpu",
        )
        radial = build_official_mace_polar_1_m_radial_gto_adapter(
            checkpoint_path=polar_checkpoint_path,
            device=device,
            long_range_evaluator_profile=(
                MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
            ),
        )
        hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
            permanent=mdp,
            response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
        )
        return hybrid, radial

    def __init__(
        self,
        device: object,
        model: str = "macemdppolarhybridddx",
        implicit: str = "none",
        solvent: str = "water",
        daily_solvent: str | None = None,
        mdp_checkpoint_path: str | Path | None = None,
        polar_checkpoint_path: str | Path | None = None,
        hessian: str = "numerical",
        d4: bool = False,
    ) -> None:
        super().__init__()
        normalized_device = str(device)
        if not normalized_device.startswith("cuda"):
            raise ValueError(
                "The hybrid ddX daily profile requires a CUDA device for "
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

        factory_solvent = str(solvent).strip() or "none"
        if daily_solvent is not None and factory_solvent.lower() != "none":
            requested_daily = str(daily_solvent).strip()
            if requested_daily.lower() != factory_solvent.lower():
                raise ValueError(
                    "Conflicting hybrid ddX solvent selectors: use either the "
                    "direct calculator solvent argument or model option "
                    "route2_solvent, not both."
                )
        requested_solvent = str(
            daily_solvent if daily_solvent is not None else factory_solvent
        ).strip()
        requested_solvent = requested_solvent or "water"
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

    def _build_pes(self, atoms: object):
        from maple.solvation.derivatives import (
            OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
            RichardsonScalarHessian,
        )
        from maple.solvation.experimental import (
            build_smd_mace_mdp_polar_hybrid_ddx_pes,
        )

        symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
        return build_smd_mace_mdp_polar_hybrid_ddx_pes(
            self._hybrid,
            symbols,
            solvent=self.solvent,
            hessian_backend=RichardsonScalarHessian(
                topology_guard_policy=OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
            ),
        )

    def _pes_for(self, atoms: object):
        symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
        if self._pes is None or self._symbols != symbols:
            self._pes = self._build_pes(atoms)
            self._symbols = symbols
        return self._pes

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
        if "forces" in requested:
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
            "profile_id": "route2-experimental-mace-mdp-polar-separated-ddx-smd-daily-v1",
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
                "forces": "experimental-same-scalar-analytic",
                "hessian": "experimental-richardson",
                "molecular_virial": "experimental-nonperiodic",
                "periodic_stress": False,
                "strict_variational_functional": False,
                "molecular_dynamics": False,
            },
            "warnings": (
                "experimental daily surface; development MAE and tail gates are "
                "not production-admitted",
                "H/FREQ uses an observed-components-only topology policy for the "
                "PySCF SMD-CDS term",
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

    def get_molecular_virial(self, atoms, *, symmetric: bool = False) -> np.ndarray:
        """Return the declared-origin molecular virial in eV.

        This is not periodic ASE stress.  The default origin is the arithmetic
        mean of the molecular coordinates, matching the underlying record.
        """

        pes = self._pes_for(atoms)
        evaluation = pes.evaluate_forces(atoms)
        virial = pes.molecular_virial(atoms, force_evaluation=evaluation)
        values = virial.symmetric_virial_eV if symmetric else virial.raw_virial_eV
        return np.array(values, dtype=float, copy=True)

    def get_hessian(self, atoms, delta: float = 0.002) -> np.ndarray:
        """Return the total same-scalar Richardson Hessian in eV/angstrom^2."""

        del delta
        evaluated = self._pes_for(atoms).evaluate_hessian(atoms)
        values = np.asarray(evaluated.hessian_eV_per_A2, dtype=float)
        expected = (3 * len(atoms), 3 * len(atoms))
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise RuntimeError(
                "Hybrid ddX Hessian is nonfinite or has the wrong shape."
            )
        return np.array(values, copy=True)

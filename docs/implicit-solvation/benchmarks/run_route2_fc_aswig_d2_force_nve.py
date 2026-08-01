#!/usr/bin/env python3
"""Short NVE gate for the D2-canonical fixed-topology Route-2 profile.

This is a deliberately *research-only* bridge around the declared total
operational scalar

    E_MACE,gas + 0.5 <c_MACE-POLAR, f_reac_CPCM> + G_fixed-topology-SMD-CDS.

The public Route-2 provider remains energy-only.  The bridge exists solely to
exercise the same analytic correction force used by the finite-difference and
closed-work canaries under a standard velocity-Verlet integrator.  It cannot
be imported as an end-user solution-phase calculator.

The protocol compares three time steps over the same short physical time and
runs a velocity-reversal check at the middle time step.  Every force evaluation
still demands a nominal SCF root and three-start local root agreement; this is
intentionally expensive, because a hidden branch jump would invalidate an NVE
claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_route2_fc_aswig_d2_force_kinematics as common
from maple.function.calculator.extra_correction.implicit.fc_aswig_smd import (
    FixedTopologyASWIGAqueousSMDImplicitSolvation,
)
from maple.function.dispatcher.md.integrator.velocity_verlet import VelocityVerlet
from maple.function.dispatcher.md.utils import (
    HA_PER_ANG_TO_AU,
    calculate_kinetic_energy,
    initialize_velocities,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "route2-fc-aswig-d2-mace-acetone-short-nve-v1"
TEMPERATURE_K = 300.0
RNG_SEED = 20260801
COMMON_PHYSICAL_TIME_FS = 0.20
TIME_STEPS_FS = (0.10, 0.05, 0.025)
REVERSAL_TIME_STEP_FS = 0.05
GATES = {
    "maximum_total_energy_drift_ev": 2.0e-5,
    "energy_drift_noise_floor_ev": 5.0e-8,
    "maximum_fine_to_coarse_drift_ratio": 0.40,
    "maximum_reversal_position_error_angstrom": 1.0e-7,
    "maximum_reversal_velocity_error_au": 1.0e-8,
    "maximum_reversal_energy_error_ev": 2.0e-5,
    "maximum_net_force_hartree_per_angstrom": 1.0e-10,
    "require_fixed_surface_cardinality": True,
    "require_nominal_roots": True,
    "require_local_multistart_agreement": True,
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-fc-aswig-d2-mace-acetone-short-nve"
    / "result.json"
)


@dataclass(frozen=True)
class _ResearchEvaluation:
    """One same-scalar force evaluation retained for the NVE audit only."""

    gas_energy_hartree: float
    correction_energy_hartree: float
    total_energy_hartree: float
    gas_force_hartree_per_angstrom: np.ndarray
    correction_force_hartree_per_angstrom: np.ndarray
    total_force_hartree_per_angstrom: np.ndarray
    surface_size: int
    scf_reason: str
    local_multistart_agreed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "gas_energy_hartree": self.gas_energy_hartree,
            "correction_energy_hartree": self.correction_energy_hartree,
            "total_energy_hartree": self.total_energy_hartree,
            "surface_size": self.surface_size,
            "scf_reason": self.scf_reason,
            "local_multistart_agreed": self.local_multistart_agreed,
            "force_sum_hartree_per_angstrom": np.sum(
                self.total_force_hartree_per_angstrom, axis=0
            ).tolist(),
        }


_RESEARCH_NVE_CAPABILITY = object()


class _Route2DirectPCMResearchNVECalculator(Calculator):
    """Private benchmark bridge; never an advertised Route-2 force API."""

    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        mace_calculator,
        solvation_options: dict[str, Any],
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _RESEARCH_NVE_CAPABILITY:
            raise PermissionError(
                "This Route-2 NVE calculator is an internal benchmark bridge, "
                "not a public force interface."
            )
        super().__init__()
        self._mace_calculator = mace_calculator
        self._solvation_options = dict(solvation_options)
        self.last_evaluation: _ResearchEvaluation | None = None
        self.evaluations: list[_ResearchEvaluation] = []

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("NVE bridge requires an explicit ASE geometry.")

        # The correction evaluator performs the complete fixed-point adjoint
        # for the direct half-coupling scalar and explicitly rejects
        # finite-resolution roots.  It intentionally remains labelled
        # research derivative evidence throughout this benchmark.
        correction = common._evaluate(
            atoms,
            self._mace_calculator,
            self._solvation_options,
            with_force=True,
        )
        root = common._nominal_root(correction)
        correction_force = np.asarray(
            correction.forces_hartree_per_angstrom,
            dtype=float,
        )
        if correction_force.shape != (len(atoms), 3) or not np.all(
            np.isfinite(correction_force)
        ):
            raise RuntimeError("Route-2 NVE correction force is invalid.")

        # Evaluate the independently declared gas scalar at the same geometry;
        # the direct PCM ledger never inserts a MACE field-energy cross term.
        gas_state, _ = self._mace_calculator.polar_state(
            atoms,
            compute_forces=True,
        )
        gas_force_ev = gas_state.fixed_field_forces_ev_per_angstrom
        if gas_force_ev is None:
            raise RuntimeError("MACE-POLAR gas force was omitted in NVE bridge.")
        gas_force = np.asarray(gas_force_ev, dtype=float) / Hartree
        if gas_force.shape != correction_force.shape or not np.all(
            np.isfinite(gas_force)
        ):
            raise RuntimeError("MACE-POLAR gas force is invalid in NVE bridge.")

        continuum = dict(correction.provenance["continuum_provider"])
        surface_size = continuum.get("surface_size")
        if not isinstance(surface_size, int) or surface_size <= 0:
            raise RuntimeError("Route-2 NVE bridge lacks a fixed surface size.")
        convergence = dict(root["scf_convergence"])
        agreement = dict(root["multi_start_root_agreement"])
        evaluation = _ResearchEvaluation(
            gas_energy_hartree=float(gas_state.energy_ev / Hartree),
            correction_energy_hartree=float(correction.energy_hartree),
            total_energy_hartree=float(
                gas_state.energy_ev / Hartree + correction.energy_hartree
            ),
            gas_force_hartree_per_angstrom=gas_force.copy(),
            correction_force_hartree_per_angstrom=correction_force.copy(),
            total_force_hartree_per_angstrom=(gas_force + correction_force).copy(),
            surface_size=surface_size,
            scf_reason=str(convergence["reason"]),
            local_multistart_agreed=bool(agreement["agreed"]),
        )
        self.last_evaluation = evaluation
        self.evaluations.append(evaluation)
        self.results = {
            # MAPLE calculators deliberately use Hartree and Hartree/Angstrom
            # at the ASE boundary; VelocityVerlet converts force units to a.u.
            "energy": evaluation.total_energy_hartree,
            "free_energy": evaluation.total_energy_hartree,
            "forces": evaluation.total_force_hartree_per_angstrom.copy(),
        }


def _make_research_nve_calculator(mace_calculator, solvation_options):
    return _Route2DirectPCMResearchNVECalculator(
        mace_calculator,
        solvation_options,
        _capability=_RESEARCH_NVE_CAPABILITY,
    )


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample(
    atoms,
    velocities_au: np.ndarray,
    bridge: _Route2DirectPCMResearchNVECalculator,
    *,
    step: int,
    timestep_fs: float,
) -> dict[str, object]:
    potential = float(atoms.get_potential_energy())
    kinetic = float(calculate_kinetic_energy(atoms, velocities_au))
    evaluation = bridge.last_evaluation
    if evaluation is None:
        raise RuntimeError("NVE energy sample has no force-evaluation provenance.")
    if abs(potential - evaluation.total_energy_hartree) > 1.0e-12:
        raise RuntimeError("NVE potential energy disagrees with its force scalar.")
    return {
        "step": int(step),
        "time_fs": float(step * timestep_fs),
        "potential_energy_hartree": potential,
        "kinetic_energy_hartree": kinetic,
        "total_energy_hartree": float(potential + kinetic),
        "evaluation": evaluation.as_dict(),
    }


def _step_count(timestep_fs: float) -> int:
    value = COMMON_PHYSICAL_TIME_FS / timestep_fs
    rounded = int(round(value))
    if rounded <= 0 or not np.isclose(value, rounded, rtol=0.0, atol=1.0e-12):
        raise ValueError("NVE time-step does not divide the frozen physical duration.")
    return rounded


def _run_trajectory(
    base,
    mace_calculator,
    solvation_options: dict[str, Any],
    initial_velocities_au: np.ndarray,
    *,
    timestep_fs: float,
    reverse_at_end: bool,
) -> dict[str, object]:
    atoms = base.copy()
    bridge = _make_research_nve_calculator(mace_calculator, solvation_options)
    atoms.calc = bridge
    integrator = VelocityVerlet(atoms, timestep_fs)
    velocities = np.asarray(initial_velocities_au, dtype=float).copy()
    initial_positions = np.asarray(atoms.positions, dtype=float).copy()
    initial_velocities = velocities.copy()
    step_count = _step_count(timestep_fs)

    forces_au = atoms.get_forces() * HA_PER_ANG_TO_AU
    samples = [_sample(atoms, velocities, bridge, step=0, timestep_fs=timestep_fs)]
    for step in range(1, step_count + 1):
        velocities, forces_au = integrator.step(velocities, forces_au)
        samples.append(
            _sample(atoms, velocities, bridge, step=step, timestep_fs=timestep_fs)
        )

    energies = np.asarray([row["total_energy_hartree"] for row in samples])
    maximum_drift_ev = float(np.max(np.abs(energies - energies[0])) * Hartree)
    result: dict[str, object] = {
        "timestep_fs": float(timestep_fs),
        "steps": step_count,
        "physical_time_fs": float(step_count * timestep_fs),
        "samples": samples,
        "maximum_absolute_total_energy_drift_ev": maximum_drift_ev,
        "final_minus_initial_total_energy_ev": float((energies[-1] - energies[0]) * Hartree),
        "surface_cardinalities": sorted(
            {int(row["evaluation"]["surface_size"]) for row in samples}
        ),
        "all_nominal_roots": bool(
            all(
                row["evaluation"]["scf_reason"]
                == "nominal-density-and-energy-v1"
                for row in samples
            )
        ),
        "all_local_multistart_agreement": bool(
            all(
                bool(row["evaluation"]["local_multistart_agreed"])
                for row in samples
            )
        ),
        "maximum_net_force_hartree_per_angstrom": float(
            max(
                np.max(
                    np.abs(
                        np.asarray(
                            row["evaluation"]["force_sum_hartree_per_angstrom"],
                            dtype=float,
                        )
                    )
                )
                for row in samples
            )
        ),
    }
    if reverse_at_end:
        reversed_velocities = -velocities
        for _ in range(step_count):
            reversed_velocities, forces_au = integrator.step(
                reversed_velocities,
                forces_au,
            )
        final_energy = float(atoms.get_potential_energy()) + float(
            calculate_kinetic_energy(atoms, reversed_velocities)
        )
        result["velocity_reversal"] = {
            "position_error_angstrom": float(
                np.max(np.abs(np.asarray(atoms.positions) - initial_positions))
            ),
            "velocity_error_au": float(
                np.max(np.abs(reversed_velocities + initial_velocities))
            ),
            "total_energy_error_ev": float((final_energy - energies[0]) * Hartree),
            "additional_force_evaluations": step_count,
        }
    return result


def _nve_gate(trajectories: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(trajectories, key=lambda row: float(row["timestep_fs"]), reverse=True)
    if tuple(float(row["timestep_fs"]) for row in ordered) != TIME_STEPS_FS:
        raise ValueError("NVE gate requires exactly the three preregistered time steps.")
    drift = [float(row["maximum_absolute_total_energy_drift_ev"]) for row in ordered]
    noise = GATES["energy_drift_noise_floor_ev"]
    ratios = [
        float(drift[index + 1] / max(drift[index], noise))
        for index in range(len(drift) - 1)
    ]
    trend_checks = [
        (
            (drift[index] <= noise and drift[index + 1] <= noise)
            or drift[index + 1]
            <= GATES["maximum_fine_to_coarse_drift_ratio"] * drift[index]
            + noise
        )
        for index in range(len(drift) - 1)
    ]
    reversal_rows = [row for row in ordered if "velocity_reversal" in row]
    if len(reversal_rows) != 1 or float(reversal_rows[0]["timestep_fs"]) != REVERSAL_TIME_STEP_FS:
        raise ValueError("NVE gate requires one reversal at the middle time step.")
    reversal = dict(reversal_rows[0]["velocity_reversal"])
    cardinalities = {
        value
        for row in ordered
        for value in list(row["surface_cardinalities"])
    }
    checks = {
        "common_physical_time": all(
            np.isclose(
                float(row["physical_time_fs"]),
                COMMON_PHYSICAL_TIME_FS,
                rtol=0.0,
                atol=1.0e-12,
            )
            for row in ordered
        ),
        "nominal_roots": all(bool(row["all_nominal_roots"]) for row in ordered),
        "local_multistart_agreement": all(
            bool(row["all_local_multistart_agreement"]) for row in ordered
        ),
        "fixed_surface_cardinality": len(cardinalities) == 1,
        "net_force": all(
            float(row["maximum_net_force_hartree_per_angstrom"])
            <= GATES["maximum_net_force_hartree_per_angstrom"]
            for row in ordered
        ),
        "absolute_energy_drift": max(drift) <= GATES["maximum_total_energy_drift_ev"],
        "second_order_drift_trend": all(trend_checks),
        "velocity_reversal_position": float(reversal["position_error_angstrom"])
        <= GATES["maximum_reversal_position_error_angstrom"],
        "velocity_reversal_velocity": float(reversal["velocity_error_au"])
        <= GATES["maximum_reversal_velocity_error_au"],
        "velocity_reversal_energy": abs(float(reversal["total_energy_error_ev"]))
        <= GATES["maximum_reversal_energy_error_ev"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "gates": dict(GATES),
        "checks": checks,
        "drift_ratios_fine_over_maximum_coarse_or_noise_floor": ratios,
        "second_order_trend_checks": [bool(value) for value in trend_checks],
        "surface_cardinalities": sorted(int(value) for value in cardinalities),
        "passed": bool(all(checks.values())),
        "remaining_force_admission_scope": (
            "this one-acetone short NVE does not by itself establish a broad "
            "public Route-2 force domain"
        ),
    }


def run(*, prepared_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    base, candidate, mol2_path, mol2_digest = common._load_atoms(prepared_path)
    settings = common._settings()
    mace_calculator = common._calculator(base, settings, device)
    initial_velocities = initialize_velocities(
        base,
        temperature=TEMPERATURE_K,
        remove_com=True,
        remove_angular=True,
        rng=np.random.default_rng(RNG_SEED),
    )
    trajectories = [
        _run_trajectory(
            base,
            mace_calculator,
            settings["solv"],
            initial_velocities,
            timestep_fs=timestep,
            reverse_at_end=(timestep == REVERSAL_TIME_STEP_FS),
        )
        for timestep in TIME_STEPS_FS
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc(),
        "git_head": common._git_head(),
        "git_worktree_status": common._git_worktree_status(),
        "runner": str(Path(__file__).relative_to(REPO_ROOT)),
        "runner_sha256": _sha256(Path(__file__)),
        "shared_kinematics_runner": str(
            Path(common.__file__).resolve().relative_to(REPO_ROOT)
        ),
        "shared_kinematics_runner_sha256": _sha256(Path(common.__file__).resolve()),
        "claim_boundary": (
            "This uses a capability-gated research bridge around the same total "
            "direct-PCM scalar and force evidence. A pass does not open a public "
            "Route-2 force API, establish a common stationary MACE--PCM electronic "
            "free-energy, prove a broad chemical force domain, or validate "
            "experimental solvation accuracy."
        ),
        "record": {
            "compound_id": common.COMPOUND_ID,
            "name": candidate["name"],
            "atom_count": len(base),
            "mol2_path": str(mol2_path),
            "mol2_sha256": mol2_digest,
        },
        "profile": common.FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        "total_declared_scalar": (
            "E_MACE-POLAR(gas)+0.5*<c_MACE-POLAR,f_reac_CPCM>+"
            "fixed-topology-aqueous-SMD-CDS"
        ),
        "temperature_kelvin": TEMPERATURE_K,
        "velocity_initialization": {
            "rng_seed": RNG_SEED,
            "remove_com": True,
            "remove_angular": True,
            "velocity_unit": "bohr/atomic-time",
        },
        "common_physical_time_fs": COMMON_PHYSICAL_TIME_FS,
        "trajectories": trajectories,
    }
    payload["short_nve_gate"] = _nve_gate(trajectories)
    _write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=common.DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args()
    if arguments.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = arguments.device
    payload = run(
        prepared_path=arguments.prepared.resolve(),
        output_path=arguments.output.resolve(),
        device=device,
    )
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build fail-closed Route-1 exact-cell analytic qualification records.

This module intentionally separates measurement records from installed runtime
admission.  A partial run is useful evidence, but it is never a passing release
artifact.  The candidate resolver is accepted only through the private factory
constructor seam used by this harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from maple.function.calculator.analytic_admission import (
    REQUIRED_ANALYTIC_CHECKS,
    REQUIRED_QUALIFICATION_GATES,
    REQUIRED_RUNTIME_DEPENDENCIES,
    runtime_implementation_paths,
)


ROOT = Path(__file__).resolve().parents[3]
GEOMETRIES = ("water", "methanol")
DIRECTION_COUNT = 3
OPERATIONS = ("energy_forces", "hessian", "hvp")
TERMINAL_STATUSES = frozenset(
    {"QUALIFIED", "FAILED", "UNAVAILABLE", "FEASIBILITY_BLOCKED", "NOT_EXECUTED"}
)
REQUIRED_GATES = REQUIRED_QUALIFICATION_GATES
REQUIRED_CHECKS = REQUIRED_ANALYTIC_CHECKS
DTYPE_GATES = {
    "D64": {
        "energy_abs_base": 2.0e-10,
        "energy_abs_scale": 1.0e-9,
        "force_max": 2.0e-7,
        "force_rms": 5.0e-8,
        "raw_hessian_asymmetry": 1.0e-10,
        "hvp_atol": 1.0e-9,
        "hvp_rtol": 1.0e-9,
        "force_fd_rms": 5.0e-6,
        "force_fd_max": 2.0e-5,
        "translation_abs": 1.0e-8,
        "translation_scale": 1.0e-8,
        "force_fd_steps_angstrom": (1.0e-3, 5.0e-4, 2.5e-4),
    },
    "N32": {
        "energy_abs": 2.0e-5,
        "force_max": 2.0e-6,
        "force_rms": 5.0e-7,
        "raw_hessian_asymmetry_scale": 2.0e-6,
        "hvp_atol": 5.0e-6,
        "hvp_rtol": 5.0e-5,
        "force_fd_rms": 5.0e-5,
        "force_fd_max": 2.0e-4,
        "translation_abs": 1.0e-5,
        "translation_scale": 1.0e-5,
        "force_fd_steps_angstrom": (2.0e-3, 1.0e-3, 5.0e-4),
    },
}


@dataclass(frozen=True)
class CellSpec:
    tier: str
    model: str
    variant: str
    device: str
    dtype_class: str
    nonpolar: str

    @property
    def profile_id(self) -> str:
        fields = (self.model, self.variant, self.device, self.dtype_class, self.nonpolar)
        return "route1-" + "-".join(value.replace(":", "_") for value in fields)


def _devices() -> tuple[str, ...]:
    """Freeze CPU plus the separately measured CUDA row, never infer availability."""
    return ("cpu", "cuda:0")


def cell_specs() -> tuple[CellSpec, ...]:
    """Return every predeclared named row; absence is represented by status."""
    models = (
        ("initial-cpu", "ani2x", "ani2x.pt", "D64"),
        ("initial-cpu", "aimnet2", "aimnet2.pt", "N32"),
        ("initial-cpu", "maceoff23m", "maceoff23m.pt", "D64"),
        ("named-mace", "maceoff23-small", "MACE-OFF23_small.model", "UNMEASURED"),
        ("named-mace", "maceoff23-medium", "MACE-OFF23_medium.model", "UNMEASURED"),
        ("named-mace", "maceoff23-large", "MACE-OFF23_large.model", "UNMEASURED"),
        ("named-mace", "maceoff24-medium", "MACE-OFF24_medium.model", "UNMEASURED"),
        ("named-mace", "mace-omol-xl", "MACE-omol-0-extra-large-1024.model", "UNMEASURED"),
        ("named-mace", "mace-polar-s", "MACEPOLAR1Smodel", "N32"),
        ("named-mace", "mace-polar-m", "MACEPOLAR1Mmodel", "N32"),
        ("named-mace", "mace-polar-l", "MACEPOLAR1Lmodel", "N32"),
        ("uma", "uma", "uma-s-1p1", "UNMEASURED"),
        ("uma", "uma", "uma-s-1p2", "UNMEASURED"),
        ("uma", "uma", "uma-m-1p1", "UNMEASURED"),
    )
    rows: list[CellSpec] = []
    for tier, model, variant, dtype_class in models:
        devices = _devices() if model != "ani2x" else ("cpu", "cuda:0")
        for device in devices:
            for nonpolar in ("ace", "none"):
                rows.append(CellSpec(tier, model, variant, device, dtype_class, nonpolar))
    return tuple(rows)


def normalized_directions(*, size: int, cell_index: int) -> tuple[tuple[int, np.ndarray], ...]:
    """Generate exactly three frozen unit directions from stable integer seeds."""
    if size <= 0 or cell_index < 0:
        raise ValueError("size must be positive and cell_index must be non-negative")
    directions = []
    for offset in range(DIRECTION_COUNT):
        seed = 730_001 + 10_007 * cell_index + offset
        vector = np.random.default_rng(seed).standard_normal(size).astype(np.float64)
        vector /= np.linalg.norm(vector)
        directions.append((seed, vector))
    return tuple(directions)


def empty_cell_record(
    spec: CellSpec,
    *,
    status: str = "NOT_EXECUTED",
    checks: Mapping[str, bool] | None = None,
    gas_profile: Mapping[str, Any] | None = None,
    solvent_profile: Mapping[str, Any] | None = None,
    dependency_versions: Mapping[str, str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    failures: Iterable[str] = (),
    evidence: Iterable[str] = (),
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Unknown cell status: {status}")
    return {
        **asdict(spec),
        "profile_id": spec.profile_id,
        "status": status,
        "gas_profile": dict(gas_profile or {}),
        "solvent_profile": dict(solvent_profile or {}),
        "operations": list(OPERATIONS),
        "checks": dict(checks or {}),
        "dependency_versions": dict(dependency_versions or {}),
        "metrics": dict(metrics or {}),
        "failures": list(failures),
        "evidence": list(evidence),
    }


def _qualified(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") == "QUALIFIED"
        and row.get("gas_profile")
        and row.get("solvent_profile")
        and row.get("dependency_versions")
        and tuple(row.get("operations", ())) == OPERATIONS
        and set(row.get("checks", ())) == set(REQUIRED_CHECKS)
        and all(row["checks"].get(name) is True for name in REQUIRED_CHECKS)
        and not row.get("failures")
    )


def compile_artifact(
    *,
    specs: Iterable[CellSpec],
    results: Mapping[str, Mapping[str, Any]],
    implementation_files_sha256: Mapping[str, str],
    gates: Mapping[str, bool],
    exhaustive_matrix_file: str,
    exhaustive_matrix_sha256: str,
    validation_files_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if len(exhaustive_matrix_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in exhaustive_matrix_sha256
    ):
        raise ValueError("A lowercase SHA256 binding to the exhaustive matrix is required")
    if Path(exhaustive_matrix_file).name != exhaustive_matrix_file or not exhaustive_matrix_file.endswith(".json"):
        raise ValueError("A simple packaged exhaustive matrix filename is required")
    specs = tuple(specs)
    if len(specs) != 1:
        raise ValueError("An installable qualification receipt must contain exactly one cell")
    cells = [dict(results.get(spec.profile_id, empty_cell_record(spec))) for spec in specs]
    all_cells_qualified = bool(cells) and all(_qualified(row) for row in cells)
    gates_passed = all(gates.get(name) is True for name in REQUIRED_GATES)
    if not all_cells_qualified:
        raise ValueError("An installable receipt requires one fully qualified exact cell")
    if not gates_passed:
        raise ValueError("An installable receipt requires every global gate")
    if not REQUIRED_RUNTIME_DEPENDENCIES.issubset(cells[0]["dependency_versions"]):
        raise ValueError("An installable receipt requires all runtime dependency versions")
    required_validation = {
        "docs/implicit-solvation/benchmarks/validate_shared_mlip_analytic.py",
        "tests/solvation/test_shared_mlip_analytic_validation.py",
    }
    if not required_validation.issubset(validation_files_sha256):
        raise ValueError("An installable receipt requires validator and test source hashes")
    return {
        "artifact_type": "route1-analytic-cell-qualification",
        "schema_version": 1,
        "passed": bool(all_cells_qualified and gates_passed),
        "all_cells_qualified": all_cells_qualified,
        "scope": "exact-qualified-cells-not-family-or-matrix",
        "exhaustive_matrix_file": exhaustive_matrix_file,
        "exhaustive_matrix_sha256": exhaustive_matrix_sha256,
        "gates": {name: gates.get(name) is True for name in REQUIRED_GATES},
        "implementation_files_sha256": dict(sorted(implementation_files_sha256.items())),
        "validation_files_sha256": dict(sorted(validation_files_sha256.items())),
        "cells": cells,
    }


def compile_matrix_report(
    *, specs: Iterable[CellSpec], results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Retain the exhaustive named matrix without turning subsets into success."""
    specs = tuple(specs)
    cells = [dict(results.get(spec.profile_id, empty_cell_record(spec))) for spec in specs]
    final = bool(cells) and all(row.get("status") != "NOT_EXECUTED" for row in cells)
    return {
        "artifact_type": "route1-analytic-cell-matrix",
        "schema_version": 1,
        "passed": bool(final and all(_qualified(row) for row in cells)),
        "final": final,
        "scope": "exhaustive-named-matrix-no-family-inference",
        "cells": cells,
    }


def candidate_factory_constructor(factory: type, resolver: Callable[..., Any]) -> Callable[..., Any]:
    """Inject only the non-user-configurable candidate admission resolver."""
    def construct(*args: Any, **kwargs: Any) -> Any:
        if "_analytic_admission_resolver" in kwargs:
            raise TypeError("candidate resolver cannot be supplied by caller")
        return factory(*args, _analytic_admission_resolver=resolver, **kwargs)

    return construct


def candidate_engine(resolver: Callable[..., Any]):
    """Create an Engine whose only altered seam is candidate admission lookup.

    InputReader/CommandControl parsing and Dispatcher execution remain the real
    public path.  The subclass copies no task behavior and does not implement a
    private workflow.
    """
    if not callable(resolver):
        raise TypeError("candidate resolver must be callable")
    from ase import Atoms
    from maple.function.calculator import SetCalculator
    from maple.function.engine import engine
    from maple.function.utility import Molecules

    class CandidateQualificationEngine(engine):
        def _mlp_initiator(self, model, device):
            atoms_for_check = None
            if isinstance(self.atoms, Atoms):
                atoms_for_check = self.atoms
            elif isinstance(self.atoms, (Molecules, list)):
                candidates = (
                    self.atoms.multiatoms if isinstance(self.atoms, Molecules) else self.atoms
                )
                if candidates:
                    atoms_for_check = candidates[0]
            factory = SetCalculator(
                device,
                model,
                self.output,
                atoms=atoms_for_check,
                d4=self.d4,
                implicit=self.commandcontrol.get("solv", {}).get("method"),
                solvent=self.commandcontrol.get("solv", {}).get("implicit"),
                model_options=self.model_options,
                solvation_options=self.commandcontrol.get("solv", {}),
                charge_options=self.commandcontrol.get("charge", {}),
                task_context={
                    "task": self.jobtype,
                    "method": self.commandcontrol.get("method"),
                    "delta": self.commandcontrol.get("delta"),
                },
                _analytic_admission_resolver=resolver,
            )
            self.calulator = factory.set_calculator()

        def _jobtype_dispatcher(self, commandcontrol, jobtype, atoms, output, extra=None):
            use_hvp = bool(
                commandcontrol.get("use_hvp", False)
                or commandcontrol.get("dimer", {}).get("use_hvp", False)
            )
            if jobtype != "ts" or commandcontrol.get("method") != "dimer" or not use_hvp:
                return super()._jobtype_dispatcher(commandcontrol, jobtype, atoms, output, extra)

            def forbidden(*_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("candidate direct HVP reached dense-H/finite-difference code")

            from unittest.mock import patch

            calculator = self.calulator
            original_gas_hessian = getattr(calculator, "_analytic_hessian", None)
            provider = calculator.solvent_correction.provider
            original_solvent_hessian = getattr(provider, "hessian", None)
            if original_gas_hessian is not None:
                calculator._analytic_hessian = forbidden
            if original_solvent_hessian is not None:
                provider.hessian = forbidden
            try:
                with patch(
                    "maple.function.calculator.calculator_base.numerical_hessian_from_atoms",
                    side_effect=forbidden,
                ):
                    result = super()._jobtype_dispatcher(
                        commandcontrol, jobtype, atoms, output, extra
                    )
            finally:
                if original_gas_hessian is not None:
                    calculator._analytic_hessian = original_gas_hessian
                if original_solvent_hessian is not None:
                    provider.hessian = original_solvent_hessian
            self.qualification_spies = {
                "runtime_finite_difference_forbidden": True,
                "direct_hvp_dense_hessian_forbidden": True,
            }
            return result

    return CandidateQualificationEngine()


def exact_candidate_resolver(
    *,
    expected_gas_profile: Mapping[str, Any],
    expected_solvent_profile: Mapping[str, Any],
    profile_id: str,
) -> Callable[[Any, Any], Any]:
    """Create a candidate resolver pinned to two predeclared complete profiles."""
    from maple.function.calculator.analytic_protocol import (
        AnalyticCompositionAdmission,
        AnalyticDerivativeAdmission,
        AnalyticIdentityError,
    )
    from maple.function.calculator.analytic_admission import cell_profile

    operations = frozenset(OPERATIONS)
    expected_gas = dict(expected_gas_profile)
    expected_solvent = dict(expected_solvent_profile)
    if not profile_id.startswith("CANDIDATE:") or not expected_gas or not expected_solvent:
        raise ValueError("candidate profile ID and both complete profiles are required")

    def resolve(gas_cell: Any, solvent_cell: Any) -> Any:
        if cell_profile(gas_cell) != expected_gas or cell_profile(solvent_cell) != expected_solvent:
            raise AnalyticIdentityError("Live cell does not match the predeclared candidate profile")
        return AnalyticCompositionAdmission(
            profile_id=profile_id,
            evidence_id=profile_id,
            gas_admission=AnalyticDerivativeAdmission(
                f"{profile_id}:gas", profile_id, gas_cell, operations
            ),
            solvent_admission=AnalyticDerivativeAdmission(
                f"{profile_id}:solvent", profile_id, solvent_cell, operations
            ),
            operations=operations,
        )

    return resolve


def _force_fd(calculator: Any, atoms: Any, step: float) -> np.ndarray:
    positions = np.asarray(atoms.positions, dtype=np.float64).copy()
    size = positions.size
    hessian = np.empty((size, size), dtype=np.float64)
    try:
        for coordinate in range(size):
            plus = positions.copy().reshape(-1)
            minus = positions.copy().reshape(-1)
            plus[coordinate] += step
            minus[coordinate] -= step
            atoms.positions[:] = plus.reshape(-1, 3)
            calculator.calculate(atoms, properties=["energy", "forces"])
            force_plus = np.asarray(calculator.results["forces"], dtype=np.float64).reshape(-1)
            atoms.positions[:] = minus.reshape(-1, 3)
            calculator.calculate(atoms, properties=["energy", "forces"])
            force_minus = np.asarray(calculator.results["forces"], dtype=np.float64).reshape(-1)
            hessian[:, coordinate] = -(force_plus - force_minus) / (2.0 * step)
    finally:
        atoms.positions[:] = positions
        calculator.calculate(atoms, properties=["energy", "forces"])
    return hessian


def measure_candidate_geometry(
    *, calculator: Any, atoms: Any, dtype_class: str, cell_index: int
) -> dict[str, Any]:
    """Measure one preregistered geometry against unchanged D64/N32 gates."""
    gate = DTYPE_GATES[dtype_class]
    calculator.calculate(atoms, properties=["energy", "forces"])
    public_energy = float(calculator.results["energy"])
    public_forces = np.asarray(calculator.results["forces"], dtype=np.float64)
    raw_hessian = np.asarray(calculator.get_hessian(atoms), dtype=np.float64)
    raw_asymmetry = float(
        np.max(np.abs(raw_hessian - raw_hessian.T), initial=0.0)
    )
    hessian_scale = float(np.max(np.abs(raw_hessian), initial=0.0))
    if dtype_class == "D64":
        energy_limit = gate["energy_abs_base"] + gate["energy_abs_scale"] * abs(public_energy)
        asymmetry_limit = gate["raw_hessian_asymmetry"]
    else:
        energy_limit = gate["energy_abs"]
        asymmetry_limit = gate["raw_hessian_asymmetry_scale"] * max(1.0, hessian_scale)

    hvp_records = []
    hvp_passed = True
    center_energy_errors = []
    center_force_maxima = []
    center_force_rms_values = []
    for seed, direction in normalized_directions(size=raw_hessian.shape[0], cell_index=cell_index):
        direct_value, center_forces, center_energy = calculator.get_hvp(atoms, direction)
        direct = np.asarray(direct_value, dtype=np.float64)
        center_forces = np.asarray(center_forces, dtype=np.float64).reshape(-1, 3)
        center_energy = float(np.asarray(center_energy))
        center_energy_errors.append(abs(center_energy - public_energy))
        center_delta = center_forces - public_forces
        center_force_maxima.append(float(np.max(np.abs(center_delta), initial=0.0)))
        center_force_rms_values.append(float(np.sqrt(np.mean(center_delta**2))))
        reference = raw_hessian @ direction
        delta = direct - reference
        passed = bool(
            np.allclose(direct, reference, atol=gate["hvp_atol"], rtol=gate["hvp_rtol"])
        )
        hvp_passed &= passed
        hvp_records.append(
            {
                "seed": seed,
                "direction": direction.tolist(),
                "max_abs": float(np.max(np.abs(delta), initial=0.0)),
                "rms": float(np.sqrt(np.mean(delta**2))),
                "passed": passed,
            }
        )
    energy_error = max(center_energy_errors)
    force_max = max(center_force_maxima)
    force_rms = max(center_force_rms_values)

    fd_records = []
    fd_matrices = []
    for step in gate["force_fd_steps_angstrom"]:
        reference = _force_fd(calculator, atoms, step)
        fd_matrices.append(reference)
        delta = raw_hessian - reference
        rms = float(np.sqrt(np.mean(delta**2)))
        maximum = float(np.max(np.abs(delta), initial=0.0))
        maximum_index = tuple(int(value) for value in np.unravel_index(np.argmax(np.abs(delta)), delta.shape))
        fd_records.append(
            {
                "step_angstrom": step,
                "rms": rms,
                "max_abs": maximum,
                "max_abs_matrix_index": list(maximum_index),
                "passed": bool(rms <= gate["force_fd_rms"] and maximum <= gate["force_fd_max"]),
            }
        )
    pair_delta = fd_matrices[-2] - fd_matrices[-1]
    pair_rms = float(np.sqrt(np.mean(pair_delta**2)))
    pair_max = float(np.max(np.abs(pair_delta), initial=0.0))
    pair_maximum_index = tuple(
        int(value) for value in np.unravel_index(np.argmax(np.abs(pair_delta)), pair_delta.shape)
    )
    pair_passed = bool(
        pair_rms <= 2.0 * gate["force_fd_rms"]
        and pair_max <= 2.0 * gate["force_fd_max"]
    )
    translations = []
    for axis in range(3):
        direction = np.zeros(raw_hessian.shape[0], dtype=np.float64)
        direction[axis::3] = 1.0
        translations.append(float(np.linalg.norm(raw_hessian @ direction)))
    translation_metric = max(translations)
    translation_limit = max(
        gate["translation_abs"], gate["translation_scale"] * np.linalg.norm(raw_hessian)
    )
    checks = {
        "center_energy": bool(energy_error <= energy_limit),
        "center_forces": bool(force_max <= gate["force_max"] and force_rms <= gate["force_rms"]),
        "raw_hessian_symmetry": bool(raw_asymmetry <= asymmetry_limit),
        "direct_hvp_vs_raw_hessian": bool(hvp_passed),
        "force_fd_finest": bool(fd_records[-1]["passed"]),
        "force_fd_second_finest": bool(fd_records[-2]["passed"]),
        "force_fd_pair_difference": bool(pair_passed),
        "translation_leakage": bool(translation_metric <= translation_limit),
    }
    return {
        "checks": checks,
        "metrics": {
            "center_energy_abs": energy_error,
            "center_energy_limit": energy_limit,
            "center_force_max": force_max,
            "center_force_rms": force_rms,
            "raw_hessian_asymmetry": raw_asymmetry,
            "raw_hessian_asymmetry_limit": asymmetry_limit,
            "translation_norms": translations,
            "translation_limit": translation_limit,
            "hvp_directions": hvp_records,
            "force_fd": fd_records,
            "force_fd_fine_pair_rms": pair_rms,
            "force_fd_fine_pair_max": pair_max,
            "force_fd_fine_pair_max_matrix_index": list(pair_maximum_index),
        },
        "passed": all(checks.values()),
    }


def implementation_paths() -> tuple[str, ...]:
    """Files whose bytes define qualification, excluding evidence and admission map.

    Runtime formula/profile data are implementation inputs and therefore remain
    included.  Only the generated installed mapping and evidence directories are
    excluded to avoid the qualification/admission hash cycle.
    """
    return runtime_implementation_paths()


def validation_paths() -> tuple[str, ...]:
    """Non-runtime harness and packaging sources retained as separate provenance."""
    paths = (
        "docs/implicit-solvation/benchmarks/validate_shared_mlip_analytic.py",
        "tests/solvation/run_no_openmm_install.py",
        "tests/solvation/test_independent_obc2_no_openmm.py",
        "tests/solvation/test_shared_mlip_analytic_validation.py",
        "pyproject.toml",
    )
    return tuple(path for path in paths if (ROOT / path).is_file())


def implementation_hashes() -> dict[str, str]:
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in implementation_paths()
    }


def validation_hashes() -> dict[str, str]:
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in validation_paths()
    }

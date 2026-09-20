from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms

from maple.function.engine import engine
from maple.function.timer import timer

pytest.importorskip("pyddx")


_SOLVATION_LINE = (
    "#solv(implicit=water,method=pb,provider=ddx,model=lpb,"
    "nonpolar=none,solvent_kappa_inverse_angstrom=0.1,experimental=true)"
)


def _run(
    tmp_path: Path,
    water_mol2: Path,
    *,
    stem: str,
    task_line: str,
    model_line: str = "#model=ani2x",
) -> tuple[engine, Path]:
    input_path = tmp_path / f"{stem}.inp"
    output_path = tmp_path / f"{stem}.out"
    input_path.write_text(
        "\n".join(
            [
                model_line,
                task_line,
                "#device=cpu",
                f"#charge(source=mol2,label=fixed-water-{stem})",
                _SOLVATION_LINE,
                "",
                "0 1",
                f"MOL2 {water_mol2}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    timer.reset()
    job = engine()
    job(str(input_path), str(output_path))
    return job, output_path


def _correction(job: engine) -> Any:
    atoms = cast(Atoms, job.atoms)
    return cast(Any, atoms.calc).solvent_correction


def _raw_force_fd_hessian(atoms: Atoms, step: float) -> np.ndarray:
    positions = atoms.get_positions().copy()
    dimension = 3 * len(atoms)
    raw = np.empty((dimension, dimension), dtype=np.float64)
    try:
        for coordinate in range(dimension):
            plus = positions.copy().reshape(-1)
            minus = positions.copy().reshape(-1)
            plus[coordinate] += step
            minus[coordinate] -= step
            atoms.set_positions(plus.reshape((-1, 3)))
            force_plus = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)
            atoms.set_positions(minus.reshape((-1, 3)))
            force_minus = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)
            raw[:, coordinate] = -(force_plus - force_minus) / (2.0 * step)
    finally:
        atoms.set_positions(positions)
    return raw


def _assert_precision(
    calculator: Any,
    *,
    effective_dtype: str,
    curvature_prepared: bool,
    recommended_step: float | None,
) -> None:
    precision = calculator.results["inference_precision"]
    assert precision["requested_dtype"] == "auto"
    assert precision["effective_dtype"] == effective_dtype
    assert precision["numerical_curvature_prepared"] is curvature_prepared
    assert precision["recommended_step_angstrom"] == recommended_step
    if curvature_prepared:
        assert precision["original_checkpoint_sha256"] == (
            "c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa"
        )
    else:
        assert precision["original_checkpoint_sha256"] is None
    assert calculator.results["solvation"]["gas_inference_precision"] == precision


def test_actual_ddlpb_sp_preserves_reference_provenance_and_frozen_inputs(
    water_mol2,
    tmp_path,
):
    job, _ = _run(tmp_path, water_mol2, stem="sp", task_line="#sp(verbose=1)")
    atoms = cast(Atoms, job.atoms)
    calculator = cast(Any, atoms.calc)
    correction = _correction(job)
    initial_charges = correction.provider.charges
    initial_radii = correction.provider.radii

    atoms.positions[1, 0] += 0.001
    energy = float(atoms.get_potential_energy(force_consistent=True))
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    solvation = calculator.results["solvation"]

    assert np.isfinite(energy)
    assert np.isfinite(forces).all()
    np.testing.assert_array_equal(correction.provider.charges, initial_charges)
    np.testing.assert_array_equal(correction.provider.radii, initial_radii)
    assert solvation["provenance"]["provider"] == "ddx"
    assert solvation["provenance"]["model"] == "lpb"
    assert solvation["provenance"]["reference_only"] is True
    assert set(solvation["components_hartree"]) == {"polar"}
    assert "delta_g_solv_hartree" not in solvation
    _assert_precision(
        calculator,
        effective_dtype="float32",
        curvature_prepared=False,
        recommended_step=None,
    )


def test_actual_ddlpb_opt_lowers_the_composed_scalar_over_two_iterations(
    water_mol2,
    tmp_path,
):
    job, output_path = _run(
        tmp_path,
        water_mol2,
        stem="opt",
        task_line="#opt(method=lbfgs,max_iter=2,max_step=0.02,verbose=1)",
    )
    comments = (tmp_path / "opt_opt_traj.xyz").read_text(encoding="utf-8")
    energies = [
        float(value) for value in re.findall(r"Energy\s*=\s*([-+0-9.eE]+)", comments)
    ]
    correction = _correction(job)
    calculator = cast(Any, cast(Atoms, job.atoms).calc)

    assert len(energies) == 3
    assert np.isfinite(energies).all()
    assert energies[-1] < energies[0]
    assert correction.provider.provenance["fixed_charge"] is True
    assert correction.provider.provenance["fixed_radius"] is True
    output_text = output_path.read_text(encoding="utf-8")
    assert "LBFGS" in output_text
    assert "ERROR:" not in output_text
    _assert_precision(
        calculator,
        effective_dtype="float32",
        curvature_prepared=False,
        recommended_step=None,
    )


def test_actual_ddlpb_mass_weighted_frequency_uses_refined_composed_force_hessian(
    water_mol2,
    tmp_path,
):
    job, output_path = _run(
        tmp_path,
        water_mol2,
        stem="freq",
        task_line="#freq(method=mw,ilowfreq=2,verbosity=1)",
        model_line="#model=ani2x(hessian=numerical)",
    )
    atoms = cast(Atoms, job.atoms)
    calculator = cast(Any, atoms.calc)
    correction = _correction(job)
    charges = correction.provider.charges
    radii = correction.provider.radii
    runtime_diagnostics = dict(calculator.last_numerical_hessian_diagnostics)
    atoms.get_potential_energy(force_consistent=True)
    runtime_precision = dict(calculator.results["inference_precision"])
    raw_h0005 = _raw_force_fd_hessian(atoms, 0.0005)
    raw_h00025 = _raw_force_fd_hessian(atoms, 0.00025)
    antisymmetry_h0005 = float(np.max(np.abs(raw_h0005 - raw_h0005.T)))
    antisymmetry_h00025 = float(np.max(np.abs(raw_h00025 - raw_h00025.T)))
    step_change_0005_00025 = float(np.max(np.abs(raw_h0005 - raw_h00025)))
    calculator.solvent_correction = None
    try:
        calculator.reset()
        raw_gas_h0005 = _raw_force_fd_hessian(atoms, 0.0005)
    finally:
        calculator.solvent_correction = correction

    assert np.isfinite(raw_h0005).all()
    assert np.isfinite(raw_h00025).all()
    assert runtime_diagnostics["cartesian_displacement_angstrom"] == pytest.approx(
        0.0005
    )
    assert runtime_diagnostics["maximum_raw_asymmetry_hartree_per_angstrom2"] <= 5.0e-4
    assert antisymmetry_h0005 <= 5.0e-4
    assert antisymmetry_h00025 <= 5.0e-4
    assert step_change_0005_00025 <= 5.0e-4
    assert np.linalg.norm(raw_h0005 - raw_gas_h0005) > 1.0e-6
    np.testing.assert_array_equal(correction.provider.charges, charges)
    np.testing.assert_array_equal(correction.provider.radii, radii)
    assert runtime_precision["effective_dtype"] == "float64"
    assert runtime_precision["numerical_curvature_prepared"] is True
    assert runtime_precision["recommended_step_angstrom"] == pytest.approx(0.0005)
    assert runtime_precision["original_checkpoint_sha256"] == (
        "c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa"
    )
    output_text = output_path.read_text(encoding="utf-8")
    assert "Starting frequency analysis calculation" in output_text
    assert "Final Gibbs free energy corr." in output_text
    assert "Frequency analysis completed" in output_text
    assert "Numerical Hessian diagnostics (before symmetrization)" in output_text
    assert "Force-FD step: 0.0005 Angstrom" in output_text
    assert "Maximum raw antisymmetry:" in output_text
    assert "Relative raw antisymmetry (Frobenius):" in output_text
    assert "effective=float64" in output_text
    assert (
        "Original checkpoint SHA256: "
        + calculator.inference_precision_provenance["original_checkpoint_sha256"]
    ) in output_text
    assert "ERROR:" not in output_text


def test_actual_ddlpb_prfo_runs_one_iteration_without_gas_only_hvp(
    water_mol2,
    tmp_path,
    monkeypatch,
):
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    def forbidden_hvp(*_args, **_kwargs):
        raise AssertionError("ddLPB PRFO must not call the gas-only HVP")

    monkeypatch.setattr(ANICalculator, "get_hvp", forbidden_hvp)
    job, output_path = _run(
        tmp_path,
        water_mol2,
        stem="prfo",
        task_line="#ts(method=prfo,max_iter=1)",
        model_line="#model=ani2x(hessian=numerical)",
    )

    atoms = cast(Atoms, job.atoms)
    calculator = cast(Any, atoms.calc)
    assert np.isfinite(atoms.get_potential_energy(force_consistent=True))
    assert np.isfinite(atoms.get_forces()).all()
    assert calculator.results["solvation"]["provenance"]["reference_only"] is True
    _assert_precision(
        calculator,
        effective_dtype="float64",
        curvature_prepared=True,
        recommended_step=0.0005,
    )
    output_text = output_path.read_text(encoding="utf-8")
    assert "Starting Transition State Search (TS) with RS-PRFO" in output_text
    assert "Numerical Hessian diagnostics (before symmetrization)" in output_text
    assert "Force-FD step: 0.0005 Angstrom" in output_text
    assert "Maximum raw antisymmetry:" in output_text
    assert "effective=float64" in output_text
    assert (
        "Original checkpoint SHA256: "
        + calculator.inference_precision_provenance["original_checkpoint_sha256"]
    ) in output_text
    assert "Maximum Iterations Reached" in output_text
    assert "ERROR:" not in output_text


def test_actual_ddlpb_dimer_runs_one_iteration_using_only_composed_forces(
    water_mol2,
    tmp_path,
    monkeypatch,
):
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    def forbidden_derivative(*_args, **_kwargs):
        raise AssertionError("ddLPB dimer must not call a gas-only HVP or full Hessian")

    monkeypatch.setattr(ANICalculator, "get_hvp", forbidden_derivative)
    monkeypatch.setattr(ANICalculator, "get_hessian", forbidden_derivative)
    job, output_path = _run(
        tmp_path,
        water_mol2,
        stem="dimer",
        task_line="#ts(method=dimer,max_iter=1,rot_max_iter=1,n_init=force)",
    )

    atoms = cast(Atoms, job.atoms)
    calculator = cast(Any, atoms.calc)
    assert np.isfinite(atoms.get_potential_energy(force_consistent=True))
    assert np.isfinite(atoms.get_forces()).all()
    assert calculator.results["solvation"]["provenance"]["provider"] == "ddx"
    _assert_precision(
        calculator,
        effective_dtype="float64",
        curvature_prepared=True,
        recommended_step=0.0005,
    )
    output_text = output_path.read_text(encoding="utf-8")
    assert "DIMER INITIAL STATE" in output_text
    assert "effective=float64" in output_text
    assert (
        "Original checkpoint SHA256: "
        + calculator.inference_precision_provenance["original_checkpoint_sha256"]
    ) in output_text
    assert (
        "Finite-difference step                   ....  0.0005 Angstrom" in output_text
    )
    assert "Dimer optimization converged at iteration" not in output_text
    assert (
        "Maximum Iterations Reached" in output_text
        or "not converged" in output_text.lower()
    )
    assert "ERROR:" not in output_text

"""Release acceptance-matrix harness for PBC-MD (WS3, production layer).

Runs a fixed set of acceptance classes against a calculator and checks each
against the pre-registered thresholds in ``validation/thresholds.toml``.  It is
designed to run backend-free with a built-in Lennard-Jones reference calculator
(a real conservative potential that also provides stress), and to accept a real
MAPLE calculator for release validation.  Each run emits a dated report and the
per-run provenance manifests written by the ensembles.

The thresholds are pre-registered (fixed before running) and versioned with the
report, so the standards are never tuned to the results.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    EV2HARTREE,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)
from .ensemble.nve import NVE
from .ensemble.nvt import NVT
from .ensemble.npt import NPT
from .evaluator import evaluate_md_properties
from .barostat.crescale import CRescaleBarostat
from .pbc import get_unwrapped_positions, wrap_positions_with_image_flags
from .provenance import collect_environment_provenance
from .units import (
    EV_PER_ANG3_TO_BAR,
    HARTREE_TO_EV,
    KELVIN_TO_HARTREE,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_THRESHOLDS = _REPO_ROOT / "validation" / "thresholds.toml"
# Loose short-run profile for the default unit layer; the production ship gate uses
# _DEFAULT_THRESHOLDS (see validation/thresholds.smoke.toml, WS-D).
_SMOKE_THRESHOLDS = _REPO_ROOT / "validation" / "thresholds.smoke.toml"


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def make_report_context(
    thresholds: Dict[str, Any],
    *,
    environment: Optional[Dict[str, Any]] = None,
    generated_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the immutable report identity before the acceptance matrix runs."""
    env = dict(environment) if environment is not None else collect_environment_provenance()
    stamp = generated_utc or _utc_stamp()
    version = thresholds.get("thresholds_version", "?")
    commit = (env.get("maple_git_commit") or "nogit")[:12]
    return {
        "artifact_id": f"md_acceptance_{stamp}_thr{version}_{commit}",
        "generated_utc": stamp,
        "environment": env,
    }


def _manifest_records(run_workdir: Optional[Path]) -> List[Dict[str, Any]]:
    if run_workdir is None:
        return []
    root = Path(run_workdir)
    if not root.exists():
        return []
    records: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*_md_manifest.json")):
        records.append({"path": str(path), "sha256": _sha256_file(path)})
    return records


def _artifact_release_criteria(
    payload: Dict[str, Any],
    *,
    production_candidate: bool,
) -> Dict[str, bool]:
    env = payload.get("environment") or {}
    summary = payload.get("summary") or {}
    return {
        "environment.maple_git_commit == git rev-parse HEAD": (
            env.get("maple_git_commit") is not None
            and env.get("maple_git_commit") == _git_head_commit()
        ),
        "environment.maple_git_dirty == false": env.get("maple_git_dirty") is False,
        "production_validation_run == true": bool(production_candidate),
        "overall == PASS": payload.get("overall") == "PASS",
        "summary.n_fail == 0": summary.get("n_fail") == 0,
        "summary.n_skip == 0": summary.get("n_skip") == 0,
        "summary.barostat_clamp_count == 0": summary.get("barostat_clamp_count") == 0,
    }


def _validation_artifact_paras(validation_artifact_id: Optional[str]) -> Dict[str, str]:
    if not validation_artifact_id:
        return {}
    return {"validation_artifact_id": validation_artifact_id}


# ──────────────────────────────────────────────────────────────────────────
# Reference calculator (backend-free, real conservative potential)
# ──────────────────────────────────────────────────────────────────────────

class MapleLJReferenceCalculator(Calculator):
    """Lennard-Jones wrapped to the MAPLE unit/capability contract.

    Energy/forces are converted eV -> Hartree (the MAPLE convention); stress is
    passed through in eV/Å³ (the MAPLE stress-unit contract).  This gives a real,
    cheap, deterministic potential with a consistent energy/stress pair for the
    acceptance matrix when no ML backend is available.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, epsilon: float = 0.0103, sigma: float = 3.40, rc: float = 6.5):
        super().__init__()
        from ase.calculators.lj import LennardJones

        self._lj = LennardJones(epsilon=epsilon, sigma=sigma, rc=rc, smooth=True)
        self.maple_model_name = "lj-reference"
        self.maple_pbc_md_supported = True
        self.maple_stress_supported = True
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT
        self.maple_stress_unit = ASE_STRESS_UNIT
        self.maple_model_options = {"epsilon_eV": epsilon, "sigma_A": sigma, "rc_A": rc}
        self.maple_neighbor_cutoff = rc

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = float(self._lj.get_potential_energy(atoms)) * EV2HARTREE
        self.results["forces"] = np.asarray(self._lj.get_forces(atoms), dtype=float) * EV2HARTREE
        if "stress" in properties:
            self.results["stress"] = np.asarray(self._lj.get_stress(atoms), dtype=float)


def lj_reference_factory() -> Callable[[], Calculator]:
    def factory() -> Calculator:
        return MapleLJReferenceCalculator()

    factory.maple_model_name = "lj-reference"  # type: ignore[attr-defined]
    factory.maple_model_options = {}  # type: ignore[attr-defined]
    return factory


def _lj_crystal(repeat: int = 3) -> Atoms:
    """Near-equilibrium FCC argon crystal sized so the LJ rc (6.5 Å) stays below the
    minimum-image radius.  repeat=3 -> 15.78 Å cell (radius 7.89 Å > 6.5); repeat=2
    (10.52 Å, radius 5.26 Å) would violate the minimum-image convention and be
    rejected by the MD neighbor-cutoff admission gate."""
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (repeat, repeat, repeat)
    return atoms


def _water_validation_box() -> Atoms:
    """Small neutral H/O periodic validation box for real ML backends.

    This is retained as the real-backend *stress* reference: it provides a
    molecular H/O chemistry surface all supported real PBC backends can evaluate,
    but it is not used for the dynamic NVE/NVT/NPT gates because unconstrained
    O-H stretches require a much smaller timestep than the LJ reference.
    """
    cell_length = 13.2  # MIC radius 6.6 Å, safely above 5–6 Å backend cutoffs.
    spacing = 5.5
    origin = 0.5 * (cell_length - spacing)
    oh1 = np.array([0.9572, 0.0, 0.0])
    oh2 = np.array([-0.2399872, 0.927297, 0.0])
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                o = np.array([origin + ix * spacing, origin + iy * spacing, origin + iz * spacing])
                symbols.extend(["O", "H", "H"])
                positions.extend([o, o + oh1, o + oh2])
    atoms = Atoms(symbols, positions=np.asarray(positions), cell=np.eye(3) * cell_length, pbc=True)
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    return atoms


def _co2_validation_box(*, cell_length: float = 10.55, spacing: float = 4.7) -> Atoms:
    """Small neutral H-free periodic CO2 box for real-backend dynamics.

    The backend-free matrix is calibrated on an LJ argon crystal.  Real MAPLE
    PBC backends must not be tested on Ar because AIMNet2 correctly rejects it
    as out-of-domain.  A compact C/O molecular box keeps the species set common
    to AIMNet2, MACE, MACE-Polar and UMA while avoiding the unconstrained O-H
    high-frequency modes that made the previous H/O dynamic gate timestep-bound
    rather than backend-bound.
    """
    # The default Ewald/PME validation cell stays inside the MIC-safe DSF scope
    # (radius 5.275 Å > 5.0 Å) while starting much closer to the AIMNet2 C/O NPT
    # working volume than the earlier 13.2 Å dilute gas box.  The old box spent
    # most of the 20 ps "production" volume-fluctuation window monotonically
    # relaxing, so Var(V) measured non-equilibrium drift rather than equilibrium
    # volume fluctuations.
    origin = 0.5 * (cell_length - spacing)
    co_bond = 1.16
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    ]
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    index = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                center = np.array(
                    [origin + ix * spacing, origin + iy * spacing, origin + iz * spacing]
                )
                axis = axes[index % len(axes)]
                index += 1
                symbols.extend(["O", "C", "O"])
                positions.extend([center - co_bond * axis, center, center + co_bond * axis])
    atoms = Atoms(symbols, positions=np.asarray(positions), cell=np.eye(3) * cell_length, pbc=True)
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    return atoms


def _uses_lj_reference_system(calc_factory) -> bool:
    model = getattr(calc_factory, "maple_model_name", None)
    if model is not None:
        return model in ("", "lj-reference")
    try:
        model = getattr(calc_factory(), "maple_model_name", None)
    except Exception:
        return True
    return model in (None, "", "lj-reference")


def _validation_coulomb_method(calc_factory) -> Optional[str]:
    options = getattr(calc_factory, "maple_model_options", None)
    if isinstance(options, dict) and options.get("coulomb"):
        return str(options["coulomb"]).lower()
    try:
        calc = calc_factory()
    except Exception:
        return None
    method = (
        getattr(calc, "lrcoulomb_method", None)
        or (getattr(calc, "maple_model_options", {}) or {}).get("coulomb")
    )
    return str(method).lower() if method else None


def _real_backend_dynamic_box(calc_factory) -> Atoms:
    method = _validation_coulomb_method(calc_factory)
    if method == "dsf":
        # DSF truncates the long-range term and has a different C/O EOS from the
        # Ewald/PME target.  Use a pre-registered DSF-conditioned box rather
        # than forcing the DSF validation through an Ewald-near-equilibrium
        # density and measuring the resulting expansion transient.
        return _co2_validation_box(cell_length=12.4, spacing=5.1)
    return _co2_validation_box()


def _validation_crystal(calc_factory, repeat: int = 3) -> Atoms:
    if _uses_lj_reference_system(calc_factory):
        return _lj_crystal(repeat)
    return _real_backend_dynamic_box(calc_factory)


def _validation_stress_reference(calc_factory, repeat: int = 3) -> Atoms:
    if _uses_lj_reference_system(calc_factory):
        return _lj_crystal(repeat)
    return _water_validation_box()


def validation_system_summary(calc_factory) -> Dict[str, Any]:
    """Machine-readable summary of the primary acceptance-matrix system."""
    dynamic = _validation_crystal(calc_factory)
    stress = _validation_stress_reference(calc_factory)

    def summarize(atoms: Atoms) -> Dict[str, Any]:
        return {
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "pbc": [bool(flag) for flag in atoms.pbc],
            "cell_A": np.asarray(atoms.cell.array, dtype=float).tolist(),
        }

    return {
        "dynamics": summarize(dynamic),
        "stress_finite_difference": summarize(stress),
    }


# ──────────────────────────────────────────────────────────────────────────
# Result model
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class AcceptanceResult:
    name: str
    status: str            # "pass" | "fail" | "skip"
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""


def load_thresholds(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else _DEFAULT_THRESHOLDS
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_smoke_thresholds() -> Dict[str, Any]:
    """Loose short-run thresholds for the default unit layer (NOT the ship gate)."""
    return load_thresholds(_SMOKE_THRESHOLDS)


def _read_thermo(path: Path) -> Dict[str, np.ndarray]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    data = np.array(rows, dtype=float)
    # Columns: Step Time Temp KE PE TE [Press Vol Press_pre Vol_pre | ...]
    return {
        "step": data[:, 0], "time": data[:, 1], "temp": data[:, 2],
        "ke": data[:, 3], "pe": data[:, 4], "te": data[:, 5],
        "raw": data,
    }


# ──────────────────────────────────────────────────────────────────────────
# Acceptance classes
# ──────────────────────────────────────────────────────────────────────────

def run_nve_energy_drift(
    calc_factory,
    thresholds,
    workdir,
    *,
    steps=300,
    timestep=0.5,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    th = thresholds["nve_energy_drift"]
    atoms = _validation_crystal(calc_factory)
    atoms.calc = calc_factory()
    out = str(workdir / "nve_drift.out")
    NVE(output=out, atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": 80.0,
        "remove_com_every": 0, "verbose": 0, "log_every": 1, "traj_every": steps,
        "rst_every": 0, "random_seed": 1,
        **_validation_artifact_paras(validation_artifact_id),
    }).run()
    thermo = _read_thermo(workdir / "nve_drift_md_thermo.dat")
    te = thermo["te"]
    e0 = te[0]
    rel_drift = float(abs(te[-1] - e0) / abs(e0)) if e0 != 0 else float("inf")
    ps = float(thermo["time"][-1] - thermo["time"][0]) / 1000.0
    abs_drift_per_atom_per_ps = (
        float(abs(te[-1] - e0)) / len(atoms) / ps if ps > 0 else float("inf")
    )
    passed = (rel_drift <= th["max_rel_drift"]) or (
        abs_drift_per_atom_per_ps <= th["max_abs_drift_ha_per_atom_per_ps"]
    )
    return AcceptanceResult(
        "nve_energy_drift", "pass" if passed else "fail", passed,
        {"rel_drift": rel_drift, "abs_drift_ha_per_atom_per_ps": abs_drift_per_atom_per_ps,
         "n_atoms": len(atoms)},
        f"rel drift {rel_drift:.2e} (<= {th['max_rel_drift']:.0e})",
    )


def run_restart_determinism(
    calc_factory,
    thresholds,
    workdir,
    *,
    timestep=0.5,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    th = thresholds["restart_determinism"]
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    def _run(tag, steps, **extra):
        atoms = _validation_crystal(calc_factory)
        atoms.calc = calc_factory()
        atoms.arrays["velocities"] = _seeded_velocities(atoms)
        NVE(output=str(workdir / f"{tag}.out"), atoms=atoms, paras={
            "steps": steps, "timestep": timestep, "init_velocities": False,
            "remove_com_every": 0, "verbose": 0, "log_every": steps,
            "traj_every": steps, "rst_every": steps,
            **_validation_artifact_paras(validation_artifact_id), **extra,
        }).run()
        return atoms

    full = _run("restart_full", 40)
    _run("restart_part1", 20)
    cont = _validation_crystal(calc_factory)
    cont.calc = calc_factory()
    NVE(output=str(workdir / "restart_part2.out"), atoms=cont, paras={
        "steps": 40, "timestep": timestep, "restart": True,
        "rst_file": str(workdir / "restart_part1_md.rst"),
        "init_velocities": False, "remove_com_every": 0, "verbose": 0,
        "log_every": 40, "traj_every": 40, "rst_every": 40,
        **_validation_artifact_paras(validation_artifact_id),
    }).run()

    dpos = float(np.max(np.abs(get_unwrapped_positions(full) - get_unwrapped_positions(cont))))
    vfull = full.arrays["velocities"]
    vcont = cont.arrays["velocities"]
    dvel = float(np.max(np.abs(vfull - vcont)) / (np.max(np.abs(vfull)) + 1e-30))
    passed = dpos <= th["position_atol_A"] and dvel <= th["velocity_rtol"]
    return AcceptanceResult(
        "restart_determinism", "pass" if passed else "fail", passed,
        {"max_position_diff_A": dpos, "max_velocity_rel_diff": dvel},
        f"20+20 vs 40: dpos {dpos:.2e} A, dvel {dvel:.2e}",
    )


def run_nvt_mean_temperature(
    calc_factory,
    thresholds,
    workdir,
    *,
    steps=6000,
    timestep=0.5,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    th = thresholds["nvt_mean_temperature"]
    target = 80.0
    atoms = _validation_crystal(calc_factory)
    atoms.calc = calc_factory()
    nvt = NVT(output=str(workdir / "nvt.out"), atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": target,
        "thermostat": "v-rescale", "tau_t": 50.0, "remove_com_every": 0,
        "verbose": 0, "log_every": 1, "traj_every": steps, "rst_every": 0,
        "random_seed": 2,
        **_validation_artifact_paras(validation_artifact_id),
    })
    nvt.run()
    thermo = _read_thermo(workdir / "nvt_md_thermo.dat")
    # Discard the initial transient (pre-registered equilibration fraction); the
    # short-run mean is transient-biased, so the run length and discard are
    # calibrated together with the block-SE gate below.
    eq_frac = float(th.get("equilibration_fraction", 0.4))
    all_temps = thermo["temp"]
    temps = all_temps[int(len(all_temps) * eq_frac):]
    mean_t = float(np.mean(temps))
    n_df = nvt._dof_policy.runtime_n_dof

    # Gate on the standard error of the *mean* temperature, not the instantaneous
    # spread.  The canonical per-sample spread is sigma_inst = T*sqrt(2/N_df); the
    # standard error of the mean is sigma_inst/sqrt(N_eff).  The thermostatted
    # series is autocorrelated (correlation time ~ tau_t), so the i.i.d.
    # sqrt(n) estimate *understates* the true error and would false-fail a
    # correct thermostat on this short run; use the block-averaged SE with the
    # i.i.d. value as a floor (the true SE is never below the i.i.d. one).
    n_blocks = int(th.get("n_blocks", 5))
    sigma_inst = target * np.sqrt(2.0 / n_df)
    sem_iid = sigma_inst / np.sqrt(len(temps))
    sem_block = _block_mean_stderr(temps, n_blocks)
    sem = max(sem_block, sem_iid) if np.isfinite(sem_block) else sem_iid
    window = float(th["k_sigma"] * sem)
    within = abs(mean_t - target) <= window
    return AcceptanceResult(
        "nvt_mean_temperature", "pass" if within else "fail", within,
        {"mean_T": mean_t, "target_T": target, "n_df": n_df, "n_samples": int(len(temps)),
         "sem_iid_K": float(sem_iid), "sem_block_K": float(sem_block),
         "sem_K": float(sem), "k_sigma_window_K": window},
        f"mean T {mean_t:.2f} K vs {target} K "
        f"(|Δ|={abs(mean_t - target):.2f} <= {window:.2f} K = {th['k_sigma']:g}·SE_mean)",
    )


def run_npt_pressure(
    calc_factory,
    thresholds,
    workdir,
    *,
    steps=200,
    timestep=0.5,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    th = thresholds["npt_pressure"]
    atoms = _validation_crystal(calc_factory)
    atoms.calc = calc_factory()
    NPT(output=str(workdir / "npt.out"), atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": 80.0, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 50.0,
        "tau_p": 1000.0, "remove_com_every": 0, "verbose": 0, "log_every": 1,
        "traj_every": steps, "rst_every": 0, "random_seed": 3,
        **_validation_artifact_paras(validation_artifact_id),
    }).run()
    thermo_text = (workdir / "npt_md_thermo.dat").read_text()
    summary_text = (workdir / "npt_md_summary.txt").read_text()
    has_primary = "Press(bar)" in thermo_text and "Vol(A^3)" in thermo_text
    has_diag = "Press_pre(bar)" in thermo_text and "Vol_pre(A^3)" in thermo_text
    summary_post = "post-rescale" in summary_text
    passed = has_primary and has_diag and (summary_post or not th["require_post_rescale_summary"])
    return AcceptanceResult(
        "npt_pressure", "pass" if passed else "fail", passed,
        {"has_primary_columns": has_primary, "has_diagnostic_columns": has_diag,
         "summary_post_rescale": summary_post},
        "post-rescale primary + pre-rescale diagnostic columns",
    )


def run_npt_com_pressure_invariance(
    calc_factory,
    thresholds,
    workdir,
    *,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    """NPT kinetic pressure must be invariant to pure COM drift when policy excludes COM.

    This is a fast white-box acceptance class for the Bernetti-Bussi c-rescale
    pressure input: a velocity field and the same field plus a pure COM boost
    must produce the same active-subspace pressure and the same deterministic
    c-rescale volume response when ``exclude_com_kinetic=True``.  The full
    kinetic pressure is also checked to move by a resolvable amount so the test
    cannot pass vacuously.
    """
    th = thresholds["npt_com_pressure_invariance"]
    atoms = _validation_crystal(calc_factory)
    atoms.calc = calc_factory()
    velocities = _seeded_velocities(atoms)
    masses = atoms.get_masses()
    total_mass = float(np.sum(masses))
    com_v = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
    velocities_internal = velocities - com_v
    boost = np.array([5.0e-5, -3.0e-5, 2.0e-5], dtype=float)
    velocities_boosted = velocities_internal + boost

    p_internal = evaluate_md_properties(
        atoms,
        need_stress=True,
        velocities_au=velocities_internal,
        exclude_com_kinetic=True,
    ).pressure_bar
    p_boosted = evaluate_md_properties(
        atoms,
        need_stress=True,
        velocities_au=velocities_boosted,
        exclude_com_kinetic=True,
    ).pressure_bar
    p_full_internal = evaluate_md_properties(
        atoms,
        need_stress=True,
        velocities_au=velocities_internal,
        exclude_com_kinetic=False,
    ).pressure_bar
    p_full_boosted = evaluate_md_properties(
        atoms,
        need_stress=True,
        velocities_au=velocities_boosted,
        exclude_com_kinetic=False,
    ).pressure_bar

    def _volume_after_barostat(velocities_au: np.ndarray) -> float:
        trial = atoms.copy()
        trial.calc = calc_factory()
        # Deterministic algebraic c-rescale check: T=0 disables stochastic
        # noise so two otherwise-identical COM-boosted pressure inputs must
        # produce exactly the same volume response.  The separate
        # npt_volume_fluctuation class exercises finite-temperature sampling.
        barostat = CRescaleBarostat(
            trial,
            pressure=1.0,
            temperature=0.0,
            tau_p=1000.0,
            timestep=0.5,
            compressibility=4.5e-5,
            rng=np.random.default_rng(101),
            exclude_com_kinetic=True,
        )
        barostat.apply(velocities_au)
        return float(trial.get_volume())

    volume_internal = _volume_after_barostat(velocities_internal)
    volume_boosted = _volume_after_barostat(velocities_boosted)

    pressure_delta = abs(float(p_boosted) - float(p_internal))
    volume_delta = abs(volume_boosted - volume_internal)
    full_shift = abs(float(p_full_boosted) - float(p_full_internal))
    passed = (
        pressure_delta <= float(th["max_abs_pressure_delta_bar"])
        and volume_delta <= float(th["max_abs_volume_delta_A3"])
        and full_shift >= float(th["min_full_com_pressure_shift_bar"])
    )
    return AcceptanceResult(
        "npt_com_pressure_invariance",
        "pass" if passed else "fail",
        passed,
        {
            "active_pressure_delta_bar": pressure_delta,
            "volume_response_delta_A3": volume_delta,
            "full_pressure_shift_bar": full_shift,
            "boost_au": boost.tolist(),
            "validation_artifact_id": validation_artifact_id,
        },
        f"COM boost leaves active pressure Δ={pressure_delta:.2e} bar and "
        f"c-rescale volume Δ={volume_delta:.2e} A^3; full-pressure shift "
        f"{full_shift:.2e} bar",
    )


def _lj_liquid(repeat: int = 3) -> Atoms:
    """A soft, low-density FCC argon cell (expanded to a=5.8 Å) used as a highly
    compressible proxy: its large equilibrium volume fluctuations make the NPT
    compressibility resolvable in a short run, and rc < the minimum-image radius.
    Named for its liquid-like fluctuation magnitude, not its (FCC) lattice.
    """
    from ase.build import bulk

    return bulk("Ar", "fcc", a=5.8, cubic=True) * (repeat, repeat, repeat)


def _validation_liquid(calc_factory) -> Atoms:
    if _uses_lj_reference_system(calc_factory):
        return _lj_liquid()
    return _real_backend_dynamic_box(calc_factory)


def _block_variance(series: np.ndarray, n_blocks: int) -> tuple:
    """Return (Var, standard error of Var) using block averaging.

    The volume series is autocorrelated, so the raw variance underestimates the
    sampling error.  ``Var`` is the full-series population variance (the physical
    <delta V^2>); the standard error is the spread of the per-block variances,
    which folds in the autocorrelation at the block scale.
    """
    series = np.asarray(series, dtype=float)
    var_total = float(np.var(series))
    block_size = max(len(series) // n_blocks, 1)
    block_vars = [
        float(np.var(series[i * block_size:(i + 1) * block_size]))
        for i in range(n_blocks)
        if len(series[i * block_size:(i + 1) * block_size]) > 1
    ]
    if len(block_vars) > 1:
        var_se = float(np.std(block_vars, ddof=1) / np.sqrt(len(block_vars)))
    else:
        var_se = float("nan")
    return var_total, var_se


def _block_mean_stderr(series: np.ndarray, n_blocks: int) -> float:
    """Standard error of the mean via block averaging (folds in autocorrelation).

    The series is split into ``n_blocks`` contiguous blocks; the standard error
    of the overall mean is the spread of the per-block means, std(means)/sqrt(n).
    When the block size exceeds the correlation time the blocks are effectively
    independent, so this is the autocorrelation-aware standard error that the
    i.i.d. sqrt(n) estimate cannot provide.  Returns NaN with fewer than two
    usable blocks (the caller falls back to the i.i.d. estimate).
    """
    series = np.asarray(series, dtype=float)
    block_size = max(len(series) // n_blocks, 1)
    means = [
        float(np.mean(series[i * block_size:(i + 1) * block_size]))
        for i in range(n_blocks)
        if len(series[i * block_size:(i + 1) * block_size]) > 0
    ]
    if len(means) > 1:
        return float(np.std(means, ddof=1) / np.sqrt(len(means)))
    return float("nan")


def _linear_drift_metrics(series: np.ndarray, timestep_fs: float) -> Dict[str, float]:
    """Return a simple stationarity diagnostic for an analyzed time series.

    ``drift_sigma`` is the fitted end-to-end linear drift over the analysis
    window divided by the window's standard deviation.  It is deliberately
    scale-free: a monotonic relaxation spanning several observed sigmas is not
    an equilibrium fluctuation window and must not be used to validate
    compressibility from Var(V).
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 2:
        return {
            "slope_per_ps": float("nan"),
            "drift_over_window": float("nan"),
            "std": float("nan"),
            "drift_sigma": float("inf"),
        }
    times_ps = np.arange(len(series), dtype=float) * float(timestep_fs) / 1000.0
    span_ps = float(times_ps[-1] - times_ps[0])
    slope = float(np.polyfit(times_ps, series, 1)[0]) if span_ps > 0.0 else float("nan")
    drift = abs(slope) * span_ps if np.isfinite(slope) else float("inf")
    std = float(np.std(series))
    if std > 0.0 and np.isfinite(std):
        drift_sigma = drift / std
    else:
        drift_sigma = 0.0 if drift == 0.0 else float("inf")
    return {
        "slope_per_ps": slope,
        "drift_over_window": drift,
        "std": std,
        "drift_sigma": float(drift_sigma),
    }


def run_npt_volume_fluctuation(
    calc_factory,
    thresholds,
    workdir,
    *,
    steps=20000,
    timestep=1.0,
    temperature=100.0,
    barostat_stride: Optional[int] = None,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    """NPT volume-fluctuation self-consistency for the production c-rescale path.

    The isothermal compressibility from the equilibrium volume fluctuations,

        kappa_fluct = Var(V) / (kB T <V>),

    is compared against the secant slope of the equation of state measured from
    two target pressures,

        kappa_eos = -(1/<V>_1) (<V>_2 - <V>_1)/(P_2 - P_1).

    Both are computed from the same c-rescale run, so this is an internally
    consistent check with no brittle frozen number: a barostat that fails to
    sample a physical volume distribution (no fluctuation, or wild fluctuation)
    drives kappa_fluct far from kappa_eos and fails the pre-registered band.

    Scope and limits (deliberate): this is a loose v1 *consistency* gate, not a
    fine barostat-type discriminator. The barostat-type guard — rejecting the
    equilibration-only Berendsen barostat that merely suppresses fluctuations —
    is enforced earlier, at NPT admission (NPT.__init__, WS4); a Berendsen run
    cannot reach this class without the explicit experimental escape hatch. The
    EOS secant sits slightly below the local kappa_T because compressibility
    stiffens with pressure, so a correct c-rescale run shows a small positive
    log10 ratio rather than zero.
    """
    th = thresholds["npt_volume_fluctuation"]
    p1, p2 = float(th["pressures_bar"][0]), float(th["pressures_bar"][1])
    eq_frac = float(th["equilibration_fraction"])
    n_blocks = int(th["n_blocks"])
    tau_p = float(th.get("tau_p_fs", 1000.0))
    stride_np = int(
        barostat_stride if barostat_stride is not None else th.get("barostat_stride", 1)
    )

    def _volume_series(tag: str, pressure: float) -> Dict[str, np.ndarray]:
        atoms = _validation_liquid(calc_factory)
        atoms.calc = calc_factory()
        NPT(output=str(workdir / f"{tag}.out"), atoms=atoms, paras={
            "steps": steps, "timestep": timestep, "temperature": temperature,
            "pressure": pressure, "thermostat": "v-rescale", "barostat": "c-rescale",
            "tau_t": 100.0, "tau_p": tau_p, "remove_com_every": 0, "verbose": 0,
            "barostat_stride": stride_np, "log_every": 1, "traj_every": steps,
            "rst_every": 0, "random_seed": 12345,
            "validation_artifact_id": validation_artifact_id or "npt_volume_fluctuation",
        }).run()
        thermo = _read_thermo(workdir / f"{tag}_md_thermo.dat")
        # Columns: Step Time Temp KE PE TE Press Vol(A^3) Press_pre Vol_pre.
        raw = thermo["raw"]
        cut = int(len(raw) * eq_frac)
        return {"volume": raw[cut:, 7], "pressure": raw[cut:, 6]}

    series1 = _volume_series("npt_vf_p1", p1)
    series2 = _volume_series("npt_vf_p2", p2)
    v1 = series1["volume"]
    v2 = series2["volume"]
    pressure1 = series1["pressure"]
    pressure2 = series2["pressure"]
    mean_v1, mean_v2 = float(np.mean(v1)), float(np.mean(v2))
    var_v1, var_se1 = _block_variance(v1, n_blocks)
    drift1 = _linear_drift_metrics(v1, timestep)
    drift2 = _linear_drift_metrics(v2, timestep)
    max_drift_sigma = float(th.get("max_volume_drift_sigma", float("inf")))
    mean_p1 = float(np.mean(pressure1))
    mean_p2 = float(np.mean(pressure2))
    sem_p1 = _block_mean_stderr(pressure1, n_blocks)
    sem_p2 = _block_mean_stderr(pressure2, n_blocks)

    # kB T in eV (K -> Hartree -> eV).
    kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV
    # Var(V)/(kT<V>) is in 1/(eV/A^3); divide by EV_PER_ANG3_TO_BAR for 1/bar.
    kappa_fluct = var_v1 / (kT_ev * mean_v1) / EV_PER_ANG3_TO_BAR
    kappa_eos = -(1.0 / mean_v1) * (mean_v2 - mean_v1) / (p2 - p1)
    rel_dv = abs(mean_v2 - mean_v1) / mean_v1

    metrics = {
        "kappa_fluct_per_bar": kappa_fluct,
        "kappa_eos_per_bar": kappa_eos,
        "mean_V1_A3": mean_v1, "mean_V2_A3": mean_v2,
        "rel_volume_change": rel_dv,
        "var_V1_A6": var_v1, "var_V1_block_se_A6": var_se1,
        "n_samples_post_eq": int(len(v1)),
        "kT_eV": kT_ev, "P1_bar": p1, "P2_bar": p2,
        "barostat_stride_NP": stride_np, "tau_p_fs": tau_p,
        "mean_P1_bar": mean_p1, "mean_P2_bar": mean_p2,
        "sem_P1_block_bar": float(sem_p1), "sem_P2_block_bar": float(sem_p2),
        "volume_slope_P1_A3_per_ps": drift1["slope_per_ps"],
        "volume_slope_P2_A3_per_ps": drift2["slope_per_ps"],
        "volume_drift_sigma_P1": drift1["drift_sigma"],
        "volume_drift_sigma_P2": drift2["drift_sigma"],
        "max_volume_drift_sigma": max_drift_sigma,
    }

    if (
        not np.isfinite(drift1["drift_sigma"])
        or not np.isfinite(drift2["drift_sigma"])
        or drift1["drift_sigma"] > max_drift_sigma
        or drift2["drift_sigma"] > max_drift_sigma
    ):
        return AcceptanceResult(
            "npt_volume_fluctuation", "fail", False, metrics,
            "non-stationary volume window: "
            f"drift {drift1['drift_sigma']:.2f}/{drift2['drift_sigma']:.2f} sigma "
            f"(max {max_drift_sigma:.2f}); fluctuation kappa would measure relaxation",
        )

    # Linear-region sanity guards (the EOS slope is a secant ~ local kappa_T only
    # in the linear regime); never let a degenerate measurement false-pass.
    if rel_dv < float(th["min_volume_change"]):
        # Inconclusive is NOT a pass: a release gate must not green on a class it
        # never actually validated.  Reported as status "skip" (distinct from a
        # physics "fail") with passed=False so the acceptance matrix flags it.
        return AcceptanceResult(
            "npt_volume_fluctuation", "skip", False, metrics,
            f"insufficient volume signal (rel change {rel_dv:.2e} < "
            f"{th['min_volume_change']}); inconclusive — widen the pressure range "
            "or use a more compressible system so kappa_T is resolvable",
        )
    if (rel_dv > float(th["max_volume_change"]) or mean_v2 >= mean_v1
            or kappa_eos <= 0.0 or not np.isfinite(kappa_fluct) or kappa_fluct <= 0.0):
        return AcceptanceResult(
            "npt_volume_fluctuation", "fail", False, metrics,
            f"non-linear/unstable response (rel change {rel_dv:.2e}, "
            f"kappa_eos {kappa_eos:.2e}/bar): EOS secant is not a valid kappa_T",
        )

    log10_ratio = float(np.log10(kappa_fluct / kappa_eos))
    metrics["log10_ratio"] = log10_ratio
    passed = abs(log10_ratio) <= float(th["log10_kappa_tol"])
    return AcceptanceResult(
        "npt_volume_fluctuation", "pass" if passed else "fail", passed, metrics,
        f"kappa_fluct {kappa_fluct:.2e} vs kappa_eos {kappa_eos:.2e} /bar "
        f"(log10 ratio {log10_ratio:+.2f}, tol {th['log10_kappa_tol']})",
    )


def run_npt_effective_energy_drift(
    calc_factory,
    thresholds,
    workdir,
    *,
    steps=4000,
    timestep=1.0,
    temperature=100.0,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    """Reversible c-rescale effective-energy diagnostic drift.

    H̃ = K + U + P_0·V − Σ ΔW_ext (thermostat + barostat + projection work) is
    logged as an integration-quality diagnostic; its residual slope measures
    finite-timestep error — the NPT analogue of the NVE energy-drift check.  A
    non-reversible/mis-scaled barostat, or incomplete work accounting, drifts H̃
    systematically even when the instantaneous pressure looks correct.  Runs on
    the compressible cell so the barostat genuinely moves the volume (a frozen
    volume would not exercise it).
    """
    th = thresholds["npt_effective_energy_drift"]
    atoms = _validation_liquid(calc_factory)
    atoms.calc = calc_factory()
    NPT(output=str(workdir / "npt_eff.out"), atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": temperature, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 100.0,
        "tau_p": 1000.0, "remove_com_every": 0, "verbose": 0, "log_every": 1,
        "traj_every": steps, "rst_every": 0, "random_seed": 2024,
        "validation_artifact_id": validation_artifact_id or "npt_effective_energy_drift",
    }).run()
    thermo = _read_thermo(workdir / "npt_eff_md_thermo.dat")
    raw = thermo["raw"]
    # Columns: Step Time Temp KE PE TE Press Vol Press_pre Vol_pre H_cons.
    if raw.shape[1] <= 10:
        return AcceptanceResult(
            "npt_effective_energy_drift", "fail", False,
            {"error": "H_cons column missing", "n_columns": int(raw.shape[1])},
            "effective-energy diagnostic column (H_cons) absent — bookkeeping did not run",
        )
    h_cons, times_fs = raw[:, 10], thermo["time"]
    cut = len(h_cons) // 5
    h, t = h_cons[cut:], times_fs[cut:]
    ps = (t[-1] - t[0]) / 1000.0 if len(t) >= 2 else 0.0
    slope = float(np.polyfit(t, h, 1)[0]) if (len(t) >= 2 and ps > 0) else float("nan")  # Ha/fs
    drift = abs(slope) * 1000.0 / len(atoms) if np.isfinite(slope) else float("inf")
    h_range = float(np.max(h) - np.min(h)) if len(h) else float("nan")
    passed = drift <= th["max_abs_drift_ha_per_atom_per_ps"]
    return AcceptanceResult(
        "npt_effective_energy_drift", "pass" if passed else "fail", passed,
        {"h_cons_drift_ha_per_atom_per_ps": drift, "h_cons_range_ha": h_range,
         "n_atoms": len(atoms), "fit_window_ps": ps},
        f"H̃ drift {drift:.2e} Ha/atom/ps "
        f"(<= {th['max_abs_drift_ha_per_atom_per_ps']:.0e})",
    )


def run_stress_finite_difference(calc_factory, thresholds, workdir, **_) -> AcceptanceResult:
    th = thresholds["stress_finite_difference"]
    base = _validation_stress_reference(calc_factory)
    # Deform the cell so the finite-difference matrix exercises both diagonal
    # stress and non-zero shear components rather than only the hydrostatic trace.
    deformation = np.array(
        [
            [1.03, 0.10, 0.04],
            [0.02, 0.97, 0.08],
            [0.05, 0.03, 1.01],
        ],
        dtype=float,
    )
    base.set_cell(np.asarray(base.cell.array) @ deformation, scale_atoms=True)
    base.calc = calc_factory()
    props = evaluate_md_properties(base, need_stress=True, velocities_au=np.zeros((len(base), 3)))
    stress = np.asarray(props.stress_ev_per_ang3, dtype=float)
    v0 = float(base.get_volume())

    components = ["xx", "yy", "zz", "yz", "xz", "xy"]
    component_mats = {
        "xx": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "yy": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "zz": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        # ASE Voigt order is [xx, yy, zz, yz, xz, xy].  For shear, the scalar
        # finite-difference parameter is the engineering strain gamma, so each
        # symmetric off-diagonal tensor entry receives gamma/2; then
        # dE/dgamma / V equals the matching Voigt shear stress.
        "yz": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
        "xz": np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        "xy": np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }

    per_component = []
    failing = []
    for index, name in enumerate(components):
        stress_ref = float(stress[index])
        per_delta = []
        for delta in th["deltas"]:
            e_plus = _energy_at_strain(base, calc_factory, component_mats[name], float(delta))
            e_minus = _energy_at_strain(base, calc_factory, component_mats[name], -float(delta))
            fd = (e_plus - e_minus) / (2.0 * float(delta) * v0) * HARTREE_TO_EV
            abs_error = abs(fd - stress_ref)
            denom = max(abs(stress_ref), float(th.get("relative_error_floor_ev_per_ang3", 1e-12)))
            rel_error = abs_error / denom
            log_error = (
                abs(np.log10(abs(fd) / abs(stress_ref)))
                if abs(stress_ref) > 0.0 and abs(fd) > 0.0 else 0.0
                if abs_error <= float(th["abs_tol_ev_per_ang3"]) else float("inf")
            )
            sign_ok = (np.sign(fd) == np.sign(stress_ref)) or abs(stress_ref) < float(th["abs_tol_ev_per_ang3"])
            abs_ok = abs_error <= float(th["abs_tol_ev_per_ang3"])
            rel_ok = rel_error <= float(th["rel_tol"])
            log_ok = log_error <= float(th["log10_magnitude_tol"])
            ok = (sign_ok or not th["require_sign_match"]) and (abs_ok or rel_ok or log_ok)
            per_delta.append(
                {
                    "delta": float(delta),
                    "fd_ev_per_ang3": float(fd),
                    "stress_ev_per_ang3": stress_ref,
                    "abs_error_ev_per_ang3": float(abs_error),
                    "rel_error": float(rel_error),
                    "log10_abs_ratio_error": float(log_error),
                    "sign_ok": bool(sign_ok),
                    "abs_ok": bool(abs_ok),
                    "rel_ok": bool(rel_ok),
                    "log_ok": bool(log_ok),
                    "ok": bool(ok),
                }
            )
        component_ok = all(d["ok"] for d in per_delta)
        if not component_ok:
            failing.append(name)
        per_component.append(
            {
                "component": name,
                "stress_ev_per_ang3": stress_ref,
                "per_delta": per_delta,
                "passed": bool(component_ok),
            }
        )

    passed = not failing
    worst = max(
        (
            (d["abs_error_ev_per_ang3"], item["component"], d)
            for item in per_component
            for d in item["per_delta"]
        ),
        key=lambda row: row[0],
    )
    fail_text = f"; failing components: {', '.join(failing)}" if failing else ""
    return AcceptanceResult(
        "stress_finite_difference", "pass" if passed else "fail", passed,
        {"components": per_component, "worst_abs_error": worst[0], "worst_component": worst[1]},
        "full Voigt stress FD [xx, yy, zz, yz, xz, xy]; "
        f"worst {worst[1]} abs error {worst[0]:.3e}, "
        f"rel {worst[2]['rel_error']:.3e}, log {worst[2]['log10_abs_ratio_error']:.3e}"
        f"{fail_text}",
    )


def run_pbc_geometry(calc_factory, thresholds, workdir, **_) -> AcceptanceResult:
    th = thresholds["pbc_geometry"]
    max_err = 0.0
    for cell in (
        np.diag([6.0, 6.0, 6.0]),
        np.array([[6.0, 0.0, 0.0], [1.0, 6.2, 0.0], [0.5, 0.7, 6.4]]),  # triclinic
    ):
        atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
        atoms.set_scaled_positions([[0.95, 1.0 - 1e-12, 1.4]])
        original = atoms.get_positions().copy()
        wrap_positions_with_image_flags(atoms)
        err = float(np.max(np.abs(get_unwrapped_positions(atoms) - original)))
        max_err = max(max_err, err)
    passed = max_err <= th["roundtrip_atol_A"]
    return AcceptanceResult(
        "pbc_geometry", "pass" if passed else "fail", passed,
        {"max_roundtrip_error_A": max_err},
        f"orthorhombic + triclinic wrap roundtrip error {max_err:.2e} A",
    )


def run_constraints_rejected(calc_factory, thresholds, workdir, **_) -> AcceptanceResult:
    atoms = _validation_crystal(calc_factory)
    atoms.calc = calc_factory()
    atoms.set_constraint(FixAtoms(indices=[0]))
    rejected = False
    try:
        NVE(output=str(workdir / "constrained.out"), atoms=atoms,
            paras={"steps": 0, "verbose": 0, "remove_com_every": 0})
    except ValueError:
        rejected = True
    passed = rejected or not thresholds["constraints"]["require_rejection"]
    return AcceptanceResult(
        "constraints_rejected", "pass" if passed else "fail", passed,
        {"rejected": rejected}, "ASE constraints + MD hard-rejected",
    )


def _seeded_velocities(atoms: Atoms) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.standard_normal((len(atoms), 3)) * 1e-4


def _energy_at_scaled_volume(base: Atoms, calc_factory, volume_factor: float) -> float:
    scaled = base.copy()
    scaled.calc = calc_factory()
    scaled.set_cell(base.cell.array * (volume_factor ** (1.0 / 3.0)), scale_atoms=True)
    return evaluate_md_properties(scaled, need_stress=False).energy_ha


def _energy_at_strain(base: Atoms, calc_factory, strain_matrix: np.ndarray, amplitude: float) -> float:
    strained = base.copy()
    strained.calc = calc_factory()
    transform = np.eye(3) + amplitude * np.asarray(strain_matrix, dtype=float)
    strained.set_cell(np.asarray(base.cell.array, dtype=float) @ transform, scale_atoms=True)
    return evaluate_md_properties(strained, need_stress=False).energy_ha


def run_barostat_clamp_free(
    calc_factory,
    thresholds,
    workdir,
    *,
    steps=200,
    timestep=0.5,
    validation_artifact_id: Optional[str] = None,
) -> AcceptanceResult:
    """A production c-rescale NPT run must not trigger the per-step stability clamp.

    A fired clamp truncates the stochastic-cell-rescaling ensemble, so for the LJ
    reference near equilibrium the clamp count must be zero.  The NPT driver makes a
    clamp fatal in production; this class confirms it never silently fired and records
    the largest log-volume excursion that had to be clipped (0 when none).
    """
    th = thresholds["barostat_clamp"]
    atoms = _validation_crystal(calc_factory)
    atoms.calc = calc_factory()
    sim = NPT(output=str(workdir / "npt_clamp.out"), atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": 80.0, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 50.0,
        "tau_p": 1000.0, "remove_com_every": 0, "verbose": 0, "log_every": steps,
        "traj_every": steps, "rst_every": 0, "random_seed": 5,
        **_validation_artifact_paras(validation_artifact_id),
    })
    sim.run()
    clamp_count = int(getattr(sim.barostat, "clamp_count", 0))
    max_excursion = float(getattr(sim.barostat, "max_abs_log_excursion", 0.0))
    max_clamps = int(th["max_clamps"])
    passed = clamp_count <= max_clamps
    return AcceptanceResult(
        "barostat_clamp_free", "pass" if passed else "fail", passed,
        {"clamp_count": clamp_count, "max_abs_log_excursion": max_excursion,
         "max_clamps": max_clamps},
        f"c-rescale clamp count {clamp_count} <= {max_clamps} "
        f"(max |Δlog V| clipped = {max_excursion:.3g})",
    )


ACCEPTANCE_CLASSES: List[Callable[..., AcceptanceResult]] = [
    run_nve_energy_drift,
    run_restart_determinism,
    run_nvt_mean_temperature,
    run_npt_pressure,
    run_npt_com_pressure_invariance,
    run_npt_volume_fluctuation,
    run_npt_effective_energy_drift,
    run_barostat_clamp_free,
    run_stress_finite_difference,
    run_pbc_geometry,
    run_constraints_rejected,
]


def run_acceptance_matrix(
    calc_factory: Callable[[], Calculator],
    thresholds: Optional[Dict[str, Any]] = None,
    workdir: Optional[Path] = None,
    *,
    quick: bool = False,
    validation_artifact_id: Optional[str] = None,
) -> List[AcceptanceResult]:
    """Run every acceptance class; return the per-class results."""
    import tempfile

    thresholds = thresholds if thresholds is not None else load_thresholds()
    cleanup = None
    if workdir is None:
        cleanup = tempfile.TemporaryDirectory()
        workdir = Path(cleanup.name)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    nve_kw = {"steps": 60} if quick else {}
    restart_kw = {}
    nvt_kw = {"steps": 120} if quick else {}
    npt_kw = {"steps": 60} if quick else {}
    npt_vf_kw = {"steps": 2000} if quick else {}
    npt_eff_kw = {"steps": 1000} if quick else {}
    real_backend = not _uses_lj_reference_system(calc_factory)
    if real_backend:
        # The restart gate is an equality check on two trajectories that should
        # see the same RST state.  For real GPU ML backends, tiny force rounding
        # differences can be amplified by the NVE stepper over 40 steps; use the
        # same conservative timestep as the NPT H-cons gate so the class remains
        # a restart-state test rather than a finite-timestep noise test.
        restart_kw = {"timestep": 0.25}
        # Real-backend dynamics use the species-safe CO2 box.  Its C/O modes do
        # not need the tiny timestep that H/O does, but real ML potentials still
        # show visible finite-step noise in the reversible NPT effective-energy
        # diagnostic at 0.125–1 fs.  Use 0.05 fs for the production
        # effective-energy gate (full window ~0.5 ps) so a failure indicates
        # backend/integrator inconsistency rather than an aggressive timestep
        # artifact; the short smoke window remains compatibility-only.
        npt_eff_kw = (
            {"steps": 1600, "timestep": 0.125}
            if quick else {"steps": 12000, "timestep": 0.05}
        )
        # The real-backend finite CO2 box is a backend compatibility gate, not a
        # long statistical production study.  12 ps retains a resolvable
        # fluctuation/EOS signal under the registered stationarity guard while
        # avoiding the late-time nonlinear relaxation seen in the 1->500 bar
        # protocol that thresholds 1.7.2 supersedes.
        npt_vf_kw = {"steps": 12000} if not quick else npt_vf_kw

    results: List[AcceptanceResult] = []
    try:
        for fn in ACCEPTANCE_CLASSES:
            kwargs = {}
            if fn is run_nve_energy_drift:
                kwargs = nve_kw
            elif fn is run_restart_determinism:
                kwargs = restart_kw
            elif fn is run_nvt_mean_temperature:
                kwargs = nvt_kw
            elif fn is run_npt_pressure:
                kwargs = npt_kw
            elif fn is run_npt_volume_fluctuation:
                kwargs = npt_vf_kw
            elif fn is run_npt_effective_energy_drift:
                kwargs = npt_eff_kw
            try:
                if validation_artifact_id is not None:
                    kwargs = {**kwargs, "validation_artifact_id": validation_artifact_id}
                results.append(fn(calc_factory, thresholds, workdir, **kwargs))
            except Exception as exc:  # a class that cannot run is recorded, not silently dropped
                results.append(AcceptanceResult(
                    fn.__name__.replace("run_", ""), "fail", False,
                    {"error": repr(exc)}, f"raised {type(exc).__name__}: {exc}",
                ))
    finally:
        if cleanup is not None:
            cleanup.cleanup()
    return results


def write_report(
    results: List[AcceptanceResult],
    thresholds: Dict[str, Any],
    outdir: Path,
    *,
    calculator_label: str = "lj-reference",
    extra: Optional[Dict[str, Any]] = None,
    artifact_id: Optional[str] = None,
    generated_utc: Optional[str] = None,
    environment: Optional[Dict[str, Any]] = None,
    run_workdir: Optional[Path] = None,
    artifact_dir: Optional[Path] = None,
) -> Path:
    """Write a dated markdown + JSON acceptance report; return the markdown path.

    ``extra`` is merged into the JSON payload (the release runner passes the
    calculator's declared unit contract and cutoff policy so the report states them).
    """
    outdir = Path(outdir)
    env = dict(environment) if environment is not None else collect_environment_provenance()
    stamp = generated_utc or _utc_stamp()
    version = thresholds.get("thresholds_version", "?")
    if artifact_id is None:
        artifact_id = make_report_context(
            thresholds, environment=env, generated_utc=stamp
        )["artifact_id"]
    artifact_root = Path(artifact_dir) if artifact_dir is not None else outdir / artifact_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    md_path = artifact_root / "report.md"
    json_path = artifact_root / "report.json"

    all_passed = all(r.passed for r in results)
    n_skip = sum(1 for r in results if r.status == "skip")
    clamp_count = next(
        (r.metrics.get("clamp_count") for r in results if r.name == "barostat_clamp_free"),
        None,
    )
    summary = {
        "all_passed": all_passed,
        "n_pass": sum(1 for r in results if r.passed),
        "n_fail": sum(1 for r in results if not r.passed and r.status != "skip"),
        "n_skip": n_skip,
        "barostat_clamp_count": clamp_count,
    }
    payload = {
        "artifact_id": artifact_id,
        "generated_utc": stamp,
        "thresholds_version": version,
        "calculator": calculator_label,
        "environment": env,
        "overall": "PASS" if all_passed else "FAIL",
        "summary": summary,
        "results": [r.__dict__ for r in results],
        "artifact_dir": str(artifact_root),
        "run_workdir": str(run_workdir) if run_workdir is not None else None,
        "manifest_files": _manifest_records(run_workdir),
        "markdown_report_path": str(md_path),
        "json_report_path": str(json_path),
    }
    if extra:
        payload.update(extra)

    production_candidate = bool(payload.pop("production_validation_candidate", False))
    criteria = _artifact_release_criteria(payload, production_candidate=production_candidate)
    failed_criteria = [name for name, passed in criteria.items() if not passed]
    payload["release_gate"] = {
        "criteria": criteria,
        "failed_criteria": failed_criteria,
        "ready": not failed_criteria,
    }
    payload["production_validated"] = payload["release_gate"]["ready"]

    lines = [
        f"# MD acceptance matrix — {payload['overall']}",
        "",
        f"- artifact: `{artifact_id}`",
        f"- generated (UTC): {stamp}",
        f"- thresholds version: {version}",
        f"- calculator: {calculator_label}",
        f"- MAPLE commit: {env.get('maple_git_commit')} (dirty={env.get('maple_git_dirty')})",
        f"- report path: `{md_path}`",
        f"- run workdir: `{payload['run_workdir']}`",
        f"- validation mode: {payload.get('validation_mode', 'production-validation')}",
        f"- production validated: {payload['production_validated']}",
        "",
        "| class | status | detail |",
        "|-------|--------|--------|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {'✅ ' + r.status if r.passed else '❌ ' + r.status} | {r.detail} |")
    md_path.write_text("\n".join(lines) + "\n")
    payload["markdown_report_sha256"] = _sha256_file(md_path)
    payload["artifact_sha256"] = {
        "report.md": payload["markdown_report_sha256"],
        "manifests": {
            record["path"]: record["sha256"] for record in payload["manifest_files"]
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return md_path

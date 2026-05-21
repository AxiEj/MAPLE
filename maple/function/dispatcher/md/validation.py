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
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
from .ensemble.nve import NVE
from .ensemble.nvt import NVT
from .ensemble.npt import NPT
from .evaluator import evaluate_md_properties
from .provenance import collect_environment_provenance
from .utils import (
    EV_PER_ANG3_TO_BAR,
    HARTREE_TO_EV,
    KELVIN_TO_HARTREE,
    get_unwrapped_positions,
    wrap_positions_with_image_flags,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_THRESHOLDS = _REPO_ROOT / "validation" / "thresholds.toml"


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
    return lambda: MapleLJReferenceCalculator()


def _lj_crystal(repeat: int = 2) -> Atoms:
    """A near-equilibrium FCC argon crystal sized so rc < the minimum-image radius."""
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (repeat, repeat, repeat)
    return atoms


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

def run_nve_energy_drift(calc_factory, thresholds, workdir, *, steps=300, timestep=0.5) -> AcceptanceResult:
    th = thresholds["nve_energy_drift"]
    atoms = _lj_crystal()
    atoms.calc = calc_factory()
    out = str(workdir / "nve_drift.out")
    NVE(output=out, atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": 80.0,
        "remove_com_every": 0, "verbose": 0, "log_every": 1, "traj_every": steps,
        "rst_every": 0, "random_seed": 1,
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


def run_restart_determinism(calc_factory, thresholds, workdir, *, timestep=0.5) -> AcceptanceResult:
    th = thresholds["restart_determinism"]

    def _run(tag, steps, **extra):
        atoms = _lj_crystal()
        atoms.calc = calc_factory()
        atoms.arrays["velocities"] = _seeded_velocities(atoms)
        NVE(output=str(workdir / f"{tag}.out"), atoms=atoms, paras={
            "steps": steps, "timestep": timestep, "init_velocities": False,
            "remove_com_every": 0, "verbose": 0, "log_every": steps,
            "traj_every": steps, "rst_every": steps, **extra,
        }).run()
        return atoms

    full = _run("restart_full", 40)
    _run("restart_part1", 20)
    cont = _lj_crystal()
    cont.calc = calc_factory()
    NVE(output=str(workdir / "restart_part2.out"), atoms=cont, paras={
        "steps": 40, "timestep": timestep, "restart": True,
        "rst_file": str(workdir / "restart_part1_md.rst"),
        "init_velocities": False, "remove_com_every": 0, "verbose": 0,
        "log_every": 40, "traj_every": 40, "rst_every": 40,
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


def run_nvt_mean_temperature(calc_factory, thresholds, workdir, *, steps=6000, timestep=0.5) -> AcceptanceResult:
    th = thresholds["nvt_mean_temperature"]
    target = 80.0
    atoms = _lj_crystal()
    atoms.calc = calc_factory()
    nvt = NVT(output=str(workdir / "nvt.out"), atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": target,
        "thermostat": "v-rescale", "tau_t": 50.0, "remove_com_every": 0,
        "verbose": 0, "log_every": 1, "traj_every": steps, "rst_every": 0,
        "random_seed": 2,
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


def run_npt_pressure(calc_factory, thresholds, workdir, *, steps=200, timestep=0.5) -> AcceptanceResult:
    th = thresholds["npt_pressure"]
    atoms = _lj_crystal()
    atoms.calc = calc_factory()
    NPT(output=str(workdir / "npt.out"), atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": 80.0, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 50.0,
        "tau_p": 1000.0, "remove_com_every": 0, "verbose": 0, "log_every": 1,
        "traj_every": steps, "rst_every": 0, "random_seed": 3,
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


def _lj_liquid(repeat: int = 3) -> Atoms:
    """A soft, low-density FCC argon cell (expanded to a=5.8 Å) used as a highly
    compressible proxy: its large equilibrium volume fluctuations make the NPT
    compressibility resolvable in a short run, and rc < the minimum-image radius.
    Named for its liquid-like fluctuation magnitude, not its (FCC) lattice.
    """
    from ase.build import bulk

    return bulk("Ar", "fcc", a=5.8, cubic=True) * (repeat, repeat, repeat)


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


def run_npt_volume_fluctuation(
    calc_factory, thresholds, workdir, *, steps=20000, timestep=1.0, temperature=100.0
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

    def _volume_series(tag: str, pressure: float) -> np.ndarray:
        atoms = _lj_liquid()
        atoms.calc = calc_factory()
        NPT(output=str(workdir / f"{tag}.out"), atoms=atoms, paras={
            "steps": steps, "timestep": timestep, "temperature": temperature,
            "pressure": pressure, "thermostat": "v-rescale", "barostat": "c-rescale",
            "tau_t": 100.0, "tau_p": 1000.0, "remove_com_every": 0, "verbose": 0,
            "log_every": 1, "traj_every": steps, "rst_every": 0, "random_seed": 12345,
            "validation_artifact_id": "npt_volume_fluctuation",
        }).run()
        thermo = _read_thermo(workdir / f"{tag}_md_thermo.dat")
        # Columns: Step Time Temp KE PE TE Press Vol(A^3) Press_pre Vol_pre.
        vol = thermo["raw"][:, 7]
        cut = int(len(vol) * eq_frac)
        return vol[cut:]

    v1 = _volume_series("npt_vf_p1", p1)
    v2 = _volume_series("npt_vf_p2", p2)
    mean_v1, mean_v2 = float(np.mean(v1)), float(np.mean(v2))
    var_v1, var_se1 = _block_variance(v1, n_blocks)

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
    }

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
    calc_factory, thresholds, workdir, *, steps=4000, timestep=1.0, temperature=100.0
) -> AcceptanceResult:
    """Reversible c-rescale conserved-quantity (effective-energy) drift.

    The NPT conserved quantity H̃ = K + U + P_0·V − Σ ΔW_ext (thermostat +
    barostat + projection work) is constant under exact dynamics; its residual
    slope measures the finite-timestep integration error — the NPT analogue of
    the NVE energy-drift check.  A non-reversible/mis-scaled barostat, or
    incomplete work accounting, drifts H̃ systematically even when the
    instantaneous pressure looks correct.  Runs on the compressible cell so the
    barostat genuinely moves the volume (a frozen volume would not exercise it).
    """
    th = thresholds["npt_effective_energy_drift"]
    atoms = _lj_liquid()
    atoms.calc = calc_factory()
    NPT(output=str(workdir / "npt_eff.out"), atoms=atoms, paras={
        "steps": steps, "timestep": timestep, "temperature": temperature, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 100.0,
        "tau_p": 1000.0, "remove_com_every": 0, "verbose": 0, "log_every": 1,
        "traj_every": steps, "rst_every": 0, "random_seed": 2024,
        "validation_artifact_id": "npt_effective_energy_drift",
    }).run()
    thermo = _read_thermo(workdir / "npt_eff_md_thermo.dat")
    raw = thermo["raw"]
    # Columns: Step Time Temp KE PE TE Press Vol Press_pre Vol_pre H_cons.
    if raw.shape[1] <= 10:
        return AcceptanceResult(
            "npt_effective_energy_drift", "fail", False,
            {"error": "H_cons column missing", "n_columns": int(raw.shape[1])},
            "conserved-energy column (H_cons) absent — bookkeeping did not run",
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
    base = _lj_crystal()
    base.calc = calc_factory()
    props = evaluate_md_properties(base, need_stress=True, velocities_au=np.zeros((len(base), 3)))
    # Configurational pressure from the stress trace (eV/Å³): P = -tr(sigma)/3.
    p_stress = -float(np.sum(props.stress_ev_per_ang3[:3])) / 3.0
    v0 = base.get_volume()

    per_delta = []
    for delta in th["deltas"]:
        e_plus = _energy_at_scaled_volume(base, calc_factory, 1.0 + delta)
        e_minus = _energy_at_scaled_volume(base, calc_factory, 1.0 - delta)
        # dE/dV in Ha/Å³ -> configurational pressure -dE/dV, converted to eV/Å³.
        dedv = (e_plus - e_minus) / ((1.0 + delta) * v0 - (1.0 - delta) * v0)
        p_fd = -dedv * HARTREE_TO_EV
        sign_ok = (np.sign(p_fd) == np.sign(p_stress)) or abs(p_stress) < 1e-12
        mag_ok = abs(p_stress) < 1e-12 or abs(np.log10(abs(p_fd) / abs(p_stress) + 1e-30)) <= th["log10_magnitude_tol"]
        per_delta.append({"delta": delta, "p_fd": p_fd, "sign_ok": bool(sign_ok), "mag_ok": bool(mag_ok)})
    passed = all(
        (d["sign_ok"] or not th["require_sign_match"]) and d["mag_ok"] for d in per_delta
    )
    return AcceptanceResult(
        "stress_finite_difference", "pass" if passed else "fail", passed,
        {"p_stress_ev_per_ang3": p_stress, "per_delta": per_delta},
        f"P_stress {p_stress:.3e} eV/A^3 vs -dE/dV finite difference",
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
    atoms = _lj_crystal()
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


ACCEPTANCE_CLASSES: List[Callable[..., AcceptanceResult]] = [
    run_nve_energy_drift,
    run_restart_determinism,
    run_nvt_mean_temperature,
    run_npt_pressure,
    run_npt_volume_fluctuation,
    run_npt_effective_energy_drift,
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
) -> List[AcceptanceResult]:
    """Run every acceptance class; return the per-class results."""
    import tempfile

    thresholds = thresholds if thresholds is not None else load_thresholds()
    cleanup = None
    if workdir is None:
        cleanup = tempfile.TemporaryDirectory()
        workdir = Path(cleanup.name)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    nve_kw = {"steps": 60} if quick else {}
    nvt_kw = {"steps": 120} if quick else {}
    npt_kw = {"steps": 60} if quick else {}
    npt_vf_kw = {"steps": 2000} if quick else {}
    npt_eff_kw = {"steps": 1000} if quick else {}

    results: List[AcceptanceResult] = []
    try:
        for fn in ACCEPTANCE_CLASSES:
            kwargs = {}
            if fn is run_nve_energy_drift:
                kwargs = nve_kw
            elif fn is run_nvt_mean_temperature:
                kwargs = nvt_kw
            elif fn is run_npt_pressure:
                kwargs = npt_kw
            elif fn is run_npt_volume_fluctuation:
                kwargs = npt_vf_kw
            elif fn is run_npt_effective_energy_drift:
                kwargs = npt_eff_kw
            try:
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
) -> Path:
    """Write a dated markdown + JSON acceptance report; return the markdown path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = collect_environment_provenance()
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = thresholds.get("thresholds_version", "?")
    commit = (env.get("maple_git_commit") or "nogit")[:12]
    artifact_id = f"md_acceptance_{stamp}_thr{version}_{commit}"

    all_passed = all(r.passed for r in results)
    payload = {
        "artifact_id": artifact_id,
        "generated_utc": stamp,
        "thresholds_version": version,
        "calculator": calculator_label,
        "environment": env,
        "overall": "PASS" if all_passed else "FAIL",
        "results": [r.__dict__ for r in results],
    }
    (outdir / f"{artifact_id}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        f"# MD acceptance matrix — {payload['overall']}",
        "",
        f"- artifact: `{artifact_id}`",
        f"- generated (UTC): {stamp}",
        f"- thresholds version: {version}",
        f"- calculator: {calculator_label}",
        f"- MAPLE commit: {env.get('maple_git_commit')} (dirty={env.get('maple_git_dirty')})",
        "",
        "| class | status | detail |",
        "|-------|--------|--------|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {'✅ ' + r.status if r.passed else '❌ ' + r.status} | {r.detail} |")
    md_path = outdir / f"{artifact_id}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path

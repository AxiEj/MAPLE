"""Current-checkout verification; does not fit or open experimental datasets."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
from ase import Atoms
import maple
from maple.function.calculator.route2._mace_mdp_polar_hybrid_ddx_calculator import (
    MACE_MDPPOLARHybridDDXCalculator,
)
from maple.function.dispatcher.dispatcher import Dispatcher

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
assert Path(maple.__file__).resolve().is_relative_to(ROOT)
RESULT = {"claim": "experimental workflow verification, not chemical accuracy",
          "stages": {}, "versions": {name: importlib.metadata.version(name)
          for name in ("torch", "mace-torch", "pyddx", "pyscf", "ase")}}


def save():
    (OUT / "real-workflows.json").write_text(
        json.dumps(RESULT, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n"
    )


def stage(name, fn):
    print("START", name, flush=True)
    start = time.monotonic()
    try:
        details = fn()
        RESULT["stages"][name] = {"status": "pass", **(details or {})}
    except Exception as exc:
        RESULT["stages"][name] = {"status": "failed", "type": type(exc).__name__, "error": str(exc)}
        traceback.print_exc()
    RESULT["stages"][name]["elapsed_s"] = time.monotonic() - start
    save()
    print("END", name, RESULT["stages"][name], flush=True)


calc = MACE_MDPPOLARHybridDDXCalculator(device="cuda", solvent="water")
atoms = Atoms("OH2", positions=[[0, 0, 0], [0.97, 0, 0], [-0.24, 0.94, 0]])
atoms.calc = calc
RESULT["initial_positions_A"] = atoms.positions.copy()
for path in (Path.home()/".cache/mace/MACE-MDP.model", Path.home()/".cache/mace/MACEPOLAR1Mmodel"):
    RESULT.setdefault("checkpoint_sha256", {})[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
for path in sorted((ROOT / "maple/solvation").rglob("*.py")):
    RESULT.setdefault("solvation_source_sha256", {})[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
save()


def sp():
    Dispatcher()(SimpleNamespace(params={}), "sp", atoms, str(OUT / "water-sp.out"))
    return {"energy_eV": atoms.get_potential_energy(), "forces_eV_per_A": atoms.get_forces(),
            "route2": calc.results["route2"]}


def opt():
    Dispatcher()(SimpleNamespace(params={"method": "lbfgs", "level": "tight", "max_iter": 20}),
                 "opt", atoms, str(OUT / "water-opt.out"))
    text = (OUT / "water-opt.out").read_text()
    RESULT["optimized_positions_A"] = atoms.positions.copy()
    forces = atoms.get_forces()
    if "LBFGS converged" not in text:
        raise RuntimeError("LBFGS did not report convergence under unchanged tight thresholds")
    return {"energy_eV": atoms.get_potential_energy(), "forces_eV_per_A": forces,
            "max_force_eV_per_A": float(np.abs(forces).max())}


def freq():
    if RESULT["stages"]["opt"]["status"] != "pass":
        raise RuntimeError("Stationary-point frequency verification requires converged OPT")
    # Instrument the existing real Hessian call; never replace it with a fixture.
    original = calc.get_hessian
    def capture(*args, **kwargs):
        hessian = original(*args, **kwargs)
        np.save(OUT / "water-hessian-eV-per-A2.npy", hessian)
        return hessian
    calc.get_hessian = capture
    try:
        Dispatcher()(SimpleNamespace(params={"method": "mw", "thermochemistry": "none", "verbose": 10}),
                     "freq", atoms, str(OUT / "water-freq.out"))
    finally:
        calc.get_hessian = original
    text = (OUT / "water-freq.out").read_text()
    summary = (OUT / "water-freq.sum").read_text()
    assert "Frequency analysis completed" in text
    assert "VIBRATIONAL ANALYSIS SUMMARY" in summary
    assert "THERMOCHEMISTRY AT" not in text and "G_corr" not in summary
    return {"hessian_file": "water-hessian-eV-per-A2.npy", "output": "water-freq.out",
            "boundary": "numerically guarded; CDS topology observation remains partial"}


stage("sp", sp)
stage("opt", opt)
stage("freq", freq)
save()

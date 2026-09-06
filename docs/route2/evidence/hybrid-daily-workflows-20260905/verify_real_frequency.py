"""Fresh, bounded-stencil FREQ replay of the converged real water geometry."""
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from maple.function.calculator.route2._mace_mdp_polar_hybrid_ddx_calculator import MACE_MDPPOLARHybridDDXCalculator
from maple.function.dispatcher.dispatcher import Dispatcher

out = Path(__file__).resolve().parent
prior = json.loads((out / "real-workflows.json").read_text())
assert prior["stages"]["opt"]["status"] == "pass"
result = {"status": "running", "geometry_source": "real-workflows.json optimized_positions_A",
          "step_reduction_limit": 6, "claim": "experimental vibrations, not thermochemistry or chemical accuracy"}
result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path("maple/function/calculator/route2/_mace_mdp_polar_hybrid_ddx_calculator.py"),
              Path("maple/function/dispatcher/frequency/frequency.py"),
              Path("maple/solvation/derivatives/scalar_finite_difference.py")]}
start = time.monotonic()
try:
    calc = MACE_MDPPOLARHybridDDXCalculator(device="cuda", solvent="water")
    atoms = Atoms("OH2", positions=prior["optimized_positions_A"])
    atoms.calc = calc
    original = calc.get_hessian
    def capture(*args, **kwargs):
        h = original(*args, **kwargs)
        np.save(out / "water-adaptive-hessian-eV-per-A2.npy", h)
        result["hessian_diagnostics"] = calc.hessian_diagnostics
        print("HESSIAN", result["hessian_diagnostics"], flush=True)
        return h
    calc.get_hessian = capture
    Dispatcher()(SimpleNamespace(params={"method": "mw", "thermochemistry": "none", "verbose": 10}),
        "freq", atoms, str(out / "water-adaptive-freq.out"))
    text = (out / "water-adaptive-freq.out").read_text()
    summary = (out / "water-adaptive-freq.sum").read_text()
    assert "Frequency analysis completed" in text
    assert "THERMOCHEMISTRY AT" not in text and "G_corr" not in summary
    result["status"] = "pass"
except Exception as exc:
    result.update(status="failed", error=str(exc), exception_type=type(exc).__name__)
    traceback.print_exc()
result["elapsed_s"] = time.monotonic() - start
(out / "real-frequency-adaptive.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
print("RESULT", result, flush=True)

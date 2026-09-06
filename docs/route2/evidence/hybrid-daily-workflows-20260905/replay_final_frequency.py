"""Final-source transform/report replay of a stored real guarded Hessian.

This is not a second checkpoint evaluation or a final-tree end-to-end MLIP run.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.frequency.frequency import MWFrequency
from maple.function.calculator.calculator_base import EV2HARTREE

out = Path(__file__).resolve().parent
prior = json.loads((out / "real-workflows.json").read_text())
audit = json.loads((out / "real-frequency-adaptive.json").read_text())
assert audit["status"] == "pass"
hpath = out / "water-adaptive-hessian-eV-per-A2.npy"
hessian = np.load(hpath, allow_pickle=False)


class RecordedRealHessian(Calculator):
    FREQUENCY_THERMOCHEMISTRY = "none"
    hessian_diagnostics = audit["hessian_diagnostics"]

    def get_hessian(self, atoms):
        np.testing.assert_array_equal(atoms.numbers, [8, 1, 1])
        np.testing.assert_array_equal(atoms.positions, prior["optimized_positions_A"])
        return hessian.copy()  # Recorded public ASE eV/Angstrom^2, not a model prediction.


atoms = Atoms("OH2", positions=prior["optimized_positions_A"])
atoms.calc = RecordedRealHessian()
Dispatcher()(SimpleNamespace(params={"method": "mw", "thermochemistry": "none", "verbose": 10}),
             "freq", atoms, str(out / "water-final-freq.out"))
text = (out / "water-final-freq.out").read_text()
summary = (out / "water-final-freq.sum").read_text()
assert "Frequency analysis completed" in text
assert "HESSIAN DIAGNOSTICS" in text
assert "partial" in text and "unobservable" in text.lower()
assert "THERMOCHEMISTRY AT" not in text and "G_corr" not in summary
job = MWFrequency(output=str(out / "water-final-mode-diagnostics.out"), atoms=atoms)
frequencies, modes = job.compute_frequencies(hessian * EV2HARTREE)
assert np.all(np.isfinite(frequencies))
assert np.count_nonzero(np.abs(frequencies) < 5.0) == 6
assert np.count_nonzero(frequencies > 5.0) == 3
assert np.count_nonzero(frequencies < -5.0) == 0
files = [Path("maple/function/dispatcher/frequency/frequency.py"),
         Path("maple/function/dispatcher/legacy_units.py"),
         Path("maple/function/dispatcher/dispatcher.py"), hpath, Path(__file__)]
result = {"status": "pass", "claim": __doc__, "frequencies_cm1": frequencies.tolist(),
          "hessian_diagnostics": audit["hessian_diagnostics"],
          "input_and_source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out / "final-frequency-replay.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

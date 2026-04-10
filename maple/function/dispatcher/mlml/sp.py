from ase import Atoms

from ..jobABC import JobABC
from maple.function.timer import timer


class MLMLSinglePoint(JobABC):
    def __init__(self, output: str, atoms: Atoms):
        super().__init__(output)
        self.atoms = atoms

    def run(self):
        with timer("MLML Single Point Energy Calculation"):
            energy = float(self.atoms.get_potential_energy())
            forces = self.atoms.get_forces()
            results = getattr(self.atoms.calc, "results", {})

            e_low_full = results.get("mlml_energy_low_full")
            e_high_core = results.get("mlml_energy_high_core")
            e_low_core = results.get("mlml_energy_low_core")
            e_corr = results.get("mlml_energy_correction")

            self.log_info(["\n", "=" * 80 + "\n", "MLML single-point summary\n", "=" * 80 + "\n"])
            self.log_info([f"E_total      : {energy:.10f} Hartree\n"])
            if e_low_full is not None:
                self.log_info([f"E_low_full   : {float(e_low_full):.10f} Hartree\n"])
            if e_high_core is not None:
                self.log_info([f"E_high_core  : {float(e_high_core):.10f} Hartree\n"])
            if e_low_core is not None:
                self.log_info([f"E_low_core   : {float(e_low_core):.10f} Hartree\n"])
            if e_corr is not None:
                self.log_info([f"E_correction : {float(e_corr):.10f} Hartree\n"])

            force_norm = float((forces**2).sum() ** 0.5)
            self.log_info([f"||F_total||  : {force_norm:.10f} Hartree/Angstrom\n"])
            self.log_info(["=" * 80 + "\n"])

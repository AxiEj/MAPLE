from ase import Atoms

from ..jobABC import JobABC
from ..md.ensemble.nve import NVE
from ..md.ensemble.nvt import NVT
from maple.function.timer import timer


class MLMLMolecularDynamics(JobABC):
    def __init__(self, output: str, atoms: Atoms, paras: dict):
        super().__init__(output)
        self.atoms = atoms
        self.paras = paras

    def run(self):
        with timer("MLML Molecular Dynamics"):
            ensemble = str(self.paras.get("ensemble", "nve")).lower()
            if ensemble == "npt":
                raise ValueError(
                    "[MLML] mlml(method=md, ensemble=npt) is not enabled yet. Phase-2 currently supports nve/nvt only."
                )

            core_count = "N/A"
            env_count = "N/A"
            calc = getattr(self.atoms, "calc", None)
            if calc is not None:
                core_indices = getattr(calc, "core_indices", None)
                env_indices = getattr(calc, "env_indices", None)
                if core_indices is not None:
                    core_count = str(len(core_indices))
                if env_indices is not None:
                    env_count = str(len(env_indices))

            self.log_info([
                "\n",
                "=" * 80 + "\n",
                "MLML molecular dynamics setup\n",
                "=" * 80 + "\n",
                "mlml_method : md\n",
                f"ensemble    : {ensemble}\n",
                f"core_atoms  : {core_count}\n",
                f"env_atoms   : {env_count}\n",
                "=" * 80 + "\n",
            ])

            if ensemble == "nve":
                md = NVE(output=self.output, atoms=self.atoms, paras=self.paras)
            elif ensemble == "nvt":
                md = NVT(output=self.output, atoms=self.atoms, paras=self.paras)
            else:
                raise ValueError(f"[MLML] MD ensemble '{ensemble}' not supported. Choose from: nve, nvt")

            md.run()

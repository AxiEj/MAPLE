from copy import deepcopy

from ase import Atoms

from ..jobABC import JobABC
from ..optimization import Optmization
from maple.function.timer import timer


class MLMLOptimization(JobABC):
    def __init__(self, output: str, atoms: Atoms, paras: dict):
        super().__init__(output)
        self.atoms = atoms
        self.paras = paras

    def run(self):
        with timer("MLML Optimization"):
            opt_method = str(self.paras.get("opt_method", "lbfgs")).lower()
            if opt_method == "rfo":
                raise ValueError(
                    "[MLML] opt_method 'rfo' is not supported in phase-1 MLML because Hessian is unavailable."
                )

            opt_params = deepcopy(self.paras)
            opt_params["method"] = opt_method

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
                "MLML optimization setup\n",
                "=" * 80 + "\n",
                "mlml_method : opt\n",
                f"opt_method  : {opt_method}\n",
                f"core_atoms  : {core_count}\n",
                f"env_atoms   : {env_count}\n",
                "=" * 80 + "\n",
            ])

            opt = Optmization(output=self.output, atoms=self.atoms, method=opt_method, params=opt_params)
            opt.run()

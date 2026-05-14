
from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer


class Optimization(JobABC):
    def __init__(self, params: dict, output: str, atoms: Atoms):
        super().__init__(output)
        self.atoms = atoms
        self.commandcontrol = params

    def run(self):
        with timer("Optimization"):
            method = str(self.commandcontrol.get('method') or 'lbfgs').lower()
            if method == 'lbfgs':
                from .algorithm import LBFGS
                return LBFGS(self.atoms, output=self.output,
                             paras=self.commandcontrol).run()
            elif method == 'rfo':
                from .algorithm import RFO
                return RFO(self.atoms, output=self.output,
                           paras=self.commandcontrol).run()
            elif method in ('sd', 'sdcg', 'cg'):
                from .algorithm import SDCG
                return SDCG(self.atoms, output=self.output,
                            paras=self.commandcontrol).run()
            else:
                raise NotImplementedError(
                    f"Unknown opt method: {method!r}. "
                    f"Supported: lbfgs, rfo, sd, sdcg, cg.")


# Deprecated misspelling kept for backward compatibility with existing
# callers (engine.py and external scripts). New code should use the
# correctly-spelled `Optimization`. Remove once all in-tree call sites
# have been migrated and a major-version bump is acceptable.
Optmization = Optimization

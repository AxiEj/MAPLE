
from ase import Atoms

from ..jobABC import JobABC

from maple.function.calculator.set_calculator import validate_pbc_capabilities
from maple.function.timer import timer


class Optimization(JobABC):
    def __init__(self, params: dict, output: str, atoms: Atoms):
        super().__init__(output)
        validate_pbc_capabilities(atoms, "opt")
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


# Deprecated misspelling: kept as a non-warning alias until in-tree callers
# are migrated to `Optimization`. New code MUST use `Optimization`; do not
# add a runtime DeprecationWarning until the call sites are cleaned up
# (warning every existing caller before the migration is noise, not signal).
Optmization = Optimization


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
                algorithm = LBFGS(self.atoms, output=self.output,
                                  paras=self.commandcontrol)
            elif method == 'rfo':
                from .algorithm import RFO
                algorithm = RFO(self.atoms, output=self.output,
                                paras=self.commandcontrol)
            elif method in ('sd', 'sdcg', 'cg'):
                from .algorithm import SDCG
                algorithm = SDCG(self.atoms, output=self.output,
                                 paras=self.commandcontrol)
            else:
                raise NotImplementedError(
                    f"Unknown opt method: {method!r}. "
                    f"Supported: lbfgs, rfo, sd, sdcg, cg.")
            from ..pure_nonmd_status import (
                is_pure_nonmd_v2, make_status, optimization_metrics,
            )
            try:
                atoms = algorithm.run()
            except Exception as exc:
                if not is_pure_nonmd_v2(self.atoms):
                    raise
                self.atoms._maple_nonmd_status = make_status(
                    workflow="opt", method=method, converged=False,
                    termination_class="execution_failure",
                    termination_reason="optimizer_execution_failed",
                    iterations=int(getattr(algorithm, "last_iterations", 0)),
                    final_metrics={"last_recorded_metrics": optimization_metrics(self.atoms)},
                    atoms=self.atoms, error_type=type(exc).__name__, error=str(exc),
                )
                return self.atoms
            if not is_pure_nonmd_v2(atoms):
                return atoms
            converged = bool(getattr(algorithm, "last_converged", False))
            status = make_status(
                workflow="opt",
                method=method,
                converged=converged,
                termination_reason=(
                    "convergence_criteria_satisfied" if converged
                    else "maximum_iterations_reached"
                ),
                iterations=int(getattr(algorithm, "last_iterations", 0)),
                final_metrics=optimization_metrics(atoms),
                atoms=atoms,
            )
            atoms._maple_nonmd_status = status
            return atoms


# Backward compatibility for the historical misspelling.
Optmization = Optimization

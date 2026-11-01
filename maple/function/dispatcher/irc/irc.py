from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer
from maple.function.dispatcher.implicit_path import (
    bind_admitted_implicit_path,
    preserve_path_calculator_caches,
)
from maple.function.dispatcher.method_registry import (
    normalize_public_method,
    validate_input_arity,
    validate_method_parameters,
)

class IRC(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='gs'):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params

    def run(self):
        with timer("IRC Calculation"):
            public_key = normalize_public_method("irc", self.method)
            validate_input_arity(public_key, 1)
            selected = validate_method_parameters(public_key, self.commandcontrol)
            self.method = public_key.value.split(".", 1)[1]
            original_positions = self.atoms.get_positions().copy()
            original_calc = self.atoms.calc
            implicit = getattr(original_calc, "solvent_correction", None) is not None
            if implicit:
                if self.method == "gs":
                    if selected.get("project_rigid_modes") is False:
                        raise ValueError("Implicit GS-IRC requires project_rigid_modes=true.")
                    selected.setdefault("project_rigid_modes", True)
                bind_admitted_implicit_path(
                    [self.atoms], self.method, require_hessian=True,
                    normalized_params=selected,
                )
            algorithms = {
                "gs": ("GS",),
                "hpc": ("HPC",),
                "lqa": ("LQA",),
                "eulerpc": ("EulerPC",),
            }
            from . import algorithm
            runner = getattr(algorithm, algorithms[self.method][0])
            if not implicit:
                return runner(
                    self.atoms, output=self.output,
                    paras={self.method: selected},
                ).run()
            working = self.atoms.copy()
            working.calc = original_calc
            with preserve_path_calculator_caches([self.atoms], enabled=True):
                irc = runner(
                    working, output=self.output,
                    paras={self.method: selected},
                )
                try:
                    return irc.run()
                finally:
                    self.atoms.set_positions(original_positions)
                    self.atoms.calc = original_calc

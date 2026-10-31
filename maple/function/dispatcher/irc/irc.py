from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer
from maple.function.dispatcher.implicit_path import (
    bind_admitted_implicit_path,
    preserve_path_calculator_caches,
    validate_path_parameters,
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
            original_positions = self.atoms.get_positions().copy()
            original_calc = self.atoms.calc
            implicit = getattr(original_calc, "solvent_correction", None) is not None
            if implicit and self.method != "gs":
                raise ValueError("Implicit IRC admits only method=gs.")
            if self.method == 'gs':
                from .algorithm import GS
                from .algorithm.gs import GSParams
                selected = validate_path_parameters("gs", self.commandcontrol, GSParams)
                if implicit:
                    if selected.get("project_rigid_modes") is False:
                        raise ValueError("Implicit GS-IRC requires project_rigid_modes=true.")
                    selected.setdefault("project_rigid_modes", True)
                    bind_admitted_implicit_path(
                        [self.atoms], "gs", require_hessian=True
                    )
                if not implicit:
                    irc = GS(self.atoms, output=self.output, paras={"gs": selected})
                    return irc.run()
                working = self.atoms.copy()
                working.calc = original_calc
                with preserve_path_calculator_caches([self.atoms], enabled=implicit):
                    irc = GS(working, output=self.output, paras={"gs": selected})
                    try:
                        return irc.run()
                    finally:
                        self.atoms.set_positions(original_positions)
                        self.atoms.calc = original_calc
            elif self.method == 'hpc':
                from .algorithm import HPC
                irc = HPC(self.atoms, output=self.output, paras=self.commandcontrol)
                return irc.run()
            elif self.method == 'eulerpc':
                from .algorithm import EulerPC
                irc = EulerPC(self.atoms, output=self.output, paras=self.commandcontrol)
                return irc.run()
            elif self.method == 'lqa':
                from .algorithm import LQA
                irc = LQA(self.atoms, output=self.output, paras=self.commandcontrol)
                return irc.run()
            else:
                raise NotImplementedError(f'IRC method {self.method} not implemented yet.')

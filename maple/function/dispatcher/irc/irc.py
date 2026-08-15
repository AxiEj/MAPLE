from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer


class IRC(JobABC):
    def __init__(
        self,
        params: dict,
        output: str,
        atoms: Atoms,
        method: str = "gs",
    ):
        super().__init__(output)
        self.atoms = atoms
        self.method = str(method).lower()
        self.output = output
        self.commandcontrol = params

    def run(self) -> dict:
        with timer("IRC Calculation"):
            if self.method == "gs":
                from .algorithm import GS

                irc = GS(self.atoms, output=self.output, paras=self.commandcontrol)
            elif self.method == "hpc":
                from .algorithm import HPC

                irc = HPC(self.atoms, output=self.output, paras=self.commandcontrol)
            elif self.method == "eulerpc":
                from .algorithm import EulerPC

                irc = EulerPC(self.atoms, output=self.output, paras=self.commandcontrol)
            elif self.method == "lqa":
                from .algorithm import LQA

                irc = LQA(self.atoms, output=self.output, paras=self.commandcontrol)
            else:
                raise NotImplementedError(
                    f"IRC method {self.method} not implemented yet."
                )
            return irc.run()

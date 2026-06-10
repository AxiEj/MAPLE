
from typing import Union

from ase import Atoms

from ..jobABC import JobABC
from ...utility import Molecules

from maple.function.timer import timer


class Optimization(JobABC):
    def __init__(self, params: dict, output: str, atoms: Union[Atoms, Molecules]):
        super().__init__(output)
        self.atoms = atoms
        self.commandcontrol = params

    def run(self):
        with timer("Optimization"):
            method = str(self.commandcontrol.get('method') or 'lbfgs').lower()

            if isinstance(self.atoms, Molecules):
                if method != 'lbfgs':
                    raise NotImplementedError(
                        f"Batch optimization of multiple structures only supports "
                        f"method=lbfgs, got {method!r}. Split the input or switch method.")
                return self._run_batch_lbfgs()

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

    def _run_batch_lbfgs(self):
        """Optimize all structures in one GPU batch with BatchLBFGS.

        The per-structure ASE calculator assigned by the engine supplies the
        loaded model; it is wrapped into the batch calculator that BatchLBFGS
        drives through prepare/get_ef_gpu/step_cart_.
        """
        from .algorithm import BatchLBFGS, LBFGSParams

        mols = self.atoms
        if not mols.multiatoms:
            return mols

        base_calc = mols.multiatoms[0].calc
        if base_calc is None:
            raise ValueError("Batch optimization requires a calculator on the input structures.")

        if type(base_calc).__name__ == 'AIMNet2Calculator':
            from ...calculator.aimnet._aimnet2_batch_calculator import AIMNet2BatchCalc
            mols.calc = AIMNet2BatchCalc.from_ase_calculator(base_calc)
        else:
            raise NotImplementedError(
                f"Batch optimization currently supports the AIMNet2 backend only; "
                f"got {type(base_calc).__name__}. Run structures one at a time.")

        params = self._init_params(LBFGSParams, self.commandcontrol, ("lbfgs", "LBFGS", "opt"))
        self.log_info([
            "\n" + "=" * 70 + "\n",
            f"Batch LBFGS Parameters ({len(mols.multiatoms)} structures)\n",
            "=" * 70 + "\n",
            f"memory:     {params.memory}\n",
            f"curvature:  {params.curvature}\n",
            f"max_step:   {params.max_step}\n",
            f"max_iter:   {params.max_iter}\n",
            f"verbose:    {params.verbose}\n",
            "=" * 70 + "\n\n",
        ])

        BatchLBFGS(
            output=self.output,
            memory=params.memory,
            curvature=params.curvature,
            maxstep=params.max_step,
            maxiter=params.max_iter,
            device=base_calc.device,
            verbose=params.verbose,
        ).run(mols)
        return mols


# Backward compatibility for the historical misspelling.
Optmization = Optimization

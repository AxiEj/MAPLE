from __future__ import annotations

from typing import cast

import numpy as np
from ase import Atoms

from maple.function.timer import timer
from maple.function.utility import Molecules

from ..jobABC import JobABC


def _validate_implicit_ts_boundary(atoms: Atoms, method: str, params: dict) -> bool:
    """Validate the shared single-geometry, composed-force TS boundary."""
    solvent_correction = getattr(
        getattr(atoms, "calc", None),
        "solvent_correction",
        None,
    )
    if solvent_correction is None:
        return False

    if method not in {"prfo", "dimer"}:
        raise ValueError(
            "The initial experimental implicit-solvent TS path supports only "
            "method=prfo or dimer."
        )
    if np.any(atoms.get_pbc()):
        raise ValueError(
            "Implicit-solvent TS is non-periodic; periodic boundary conditions "
            "are not supported."
        )

    charge_mode = getattr(solvent_correction, "mode", None)
    if charge_mode is None or str(charge_mode).lower() != "fixed":
        raise ValueError(
            "Route 1 implicit-solvent TS requires the correction to declare "
            "mode=fixed; polarizable or undeclared charge response is outside "
            "the initial experimental contract."
        )
    if getattr(solvent_correction, "inner_mode", None) is not None:
        raise ValueError(
            "The experimental single-geometry implicit-solvent TS path supports a "
            "single solute geometry without inner=prebuilt."
        )
    supported = set(getattr(solvent_correction, "supported_properties", {"energy"}))
    if "forces" not in supported:
        raise ValueError(
            "Implicit-solvent TS requires an energy-consistent solvent "
            "correction with force support."
        )

    calculator = atoms.calc
    if getattr(calculator, "SUPPORTS_IMPLICIT_SOLVATION", False) is not True:
        raise ValueError(
            "The calculator must declare SUPPORTS_IMPLICIT_SOLVATION=True before "
            "it can enter the Route 1 TS composition path."
        )
    if method == "prfo" and str(getattr(calculator, "hessian", "")).lower() != "numerical":
        raise ValueError(
            "Implicit-solvent PRFO requires hessian=numerical so the Hessian "
            "differentiates the complete composed force."
        )
    if method == "dimer":
        dimer_params = JobABC._select_subdict(params, ("dimer", "ts"))
        if dimer_params.get("use_hvp") is True:
            raise ValueError(
                "Implicit-solvent dimer requires composed-force finite differences; "
                "gas-only autograd HVP is not supported."
            )
    return True


class TransitionState(JobABC):

    def __init__(
        self,
        output: str,
        atoms: Atoms | Molecules | list[Atoms],
        params: dict,
        method: str | None = None,
    ):
        super().__init__(output)
        self.atoms = atoms
        self.params = params
        self.method = method

    def run(self):

        with timer("Transition State Optimization"):
            if self.method is None:
                raise ValueError('Method is not provided.')

            implicit_ts = False
            if isinstance(self.atoms, Atoms):
                implicit_ts = _validate_implicit_ts_boundary(
                    self.atoms,
                    str(self.method).lower(),
                    self.params,
                )
            else:
                structures = (
                    self.atoms.multiatoms
                    if isinstance(self.atoms, Molecules)
                    else self.atoms
                )
                if any(
                    getattr(getattr(atoms, "calc", None), "solvent_correction", None)
                    is not None
                    for atoms in structures
                ):
                    raise ValueError(
                        "The initial experimental implicit-solvent TS path requires "
                        "exactly one Atoms geometry and method=prfo or dimer."
                    )
            if implicit_ts:
                derivative = (
                    "numerical Hessian"
                    if self.method == "prfo"
                    else "finite-difference curvature from the complete composed forces"
                )
                self.log_info(
                    [
                        f"\nEXPERIMENTAL IMPLICIT-SOLVENT {self.method.upper()} BOUNDARY\n",
                        (
                            "This single-geometry workflow uses the complete "
                            "composed MLIP+implicit-solvent energy, force, and "
                            f"{derivative}. It is a TS candidate-search "
                            "capability, not a claim of convergence or accuracy "
                            "certification.\n"
                        ),
                    ]
                )

            if self.method == 'prfo':
                from .algorithm import PRFO
                prfo = PRFO(
                    atoms=cast(Atoms, self.atoms),
                    output=self.output,
                    paras=self.params,
                )
                prfo.run()
                
            elif self.method == 'neb':
                # NEB now accepts Molecules object
                if isinstance(self.atoms, Molecules):
                    from .algorithm import NEB
                    neb = NEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    neb.run()
                elif isinstance(self.atoms, list):
                    # Legacy support for list input
                    if len(self.atoms) < 2:
                        raise ValueError('For NEB method, you should provide at least two structures (initial and final states).')
                    from .algorithm import NEB
                    # Convert list to Molecules for NEB
                    molecules = Molecules(self.atoms)
                    neb = NEB(
                        output=self.output,
                        atoms_or_molecules=molecules,
                        paras=self.params
                    )
                    neb.run()
                else:
                    raise ValueError('For NEB method, you should provide a Molecules object or a list of at least two structures.')
                    
            elif self.method == 'string':
                # TODO: Update String/GSM to accept Molecules object instead of list
                if not isinstance(self.atoms, list):
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                if len(self.atoms) < 2:
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                from .algorithm import GSM
                string = GSM(
                    output=self.output,
                    atoms_R=self.atoms[0],
                    atoms_P=self.atoms[1],
                    paras=self.params
                )
                string.run()
                
            elif self.method == 'dimer':
                from .algorithm import Dimer
                # Dimer takes a single Atoms as TS guess
                if isinstance(self.atoms, list):
                    atoms_input = self.atoms[0]
                else:
                    atoms_input = self.atoms
                dimer = Dimer(
                    output=self.output,
                    atoms_init=cast(Atoms, atoms_input),
                    paras=self.params
                )
                dimer.run()
                
            elif self.method == 'autoneb':
                # AutoNEB: automated multi-step reaction pathway exploration
                if isinstance(self.atoms, Molecules):
                    from .algorithm import AutoNEB
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    autoneb.run()
                elif isinstance(self.atoms, list):
                    if len(self.atoms) < 2:
                        raise ValueError('For AutoNEB method, you should provide at least two structures.')
                    from .algorithm import AutoNEB
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    autoneb.run()
                else:
                    raise ValueError('For AutoNEB method, you should provide a Molecules object or a list of structures.')

            else:
                raise ValueError(f'Method {self.method} not recognized. Available methods are: prfo, neb, string, dimer, autoneb.')

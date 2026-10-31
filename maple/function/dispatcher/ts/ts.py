from __future__ import annotations

from typing import cast

import numpy as np
from ase import Atoms

from maple.function.timer import timer
from maple.function.utility import Molecules
from maple.function.dispatcher.implicit_path import (
    bind_admitted_implicit_path,
    path_images,
    preserve_path_calculator_caches,
    validate_path_parameters,
)

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
    hessian_mode = str(getattr(calculator, "hessian", "")).lower()
    if method == "prfo":
        if hessian_mode == "numerical":
            pass
        elif (
            hessian_mode == "analytic"
            and getattr(
                calculator, "analytic_implicit_derivatives_admitted", False
            )
            is True
            and getattr(
                solvent_correction, "analytic_task_derivatives_admitted", False
            )
            is True
            and callable(getattr(solvent_correction, "get_hessian", None))
        ):
            pass
        elif hessian_mode == "analytic":
            raise ValueError(
                "Implicit-solvent analytic PRFO requires an admitted analytic "
                "solvent Hessian."
            )
        else:
            raise ValueError(
                "Implicit-solvent PRFO requires hessian=numerical or an admitted "
                "hessian=analytic solvent derivative backend."
            )
    if method == "dimer":
        dimer_params = JobABC._select_subdict(params, ("dimer", "ts"))
        if dimer_params.get("use_hvp") is True:
            if not (
                hessian_mode == "analytic"
                and getattr(
                    calculator, "analytic_implicit_derivatives_admitted", False
                )
                is True
                and getattr(
                    solvent_correction,
                    "analytic_task_derivatives_admitted",
                    False,
                )
                is True
                and callable(
                    getattr(
                        solvent_correction, "get_directional_derivatives", None
                    )
                )
            ):
                raise ValueError(
                    "Implicit-solvent Dimer use_hvp=true requires analytic gas and "
                    "solvent directional derivatives from the same composed potential."
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
            path_structures = None
            if isinstance(self.atoms, Atoms):
                implicit_ts = _validate_implicit_ts_boundary(
                    self.atoms,
                    str(self.method).lower(),
                    self.params,
                )
            else:
                structures = path_images(self.atoms)
                self.atoms = structures
                path_structures = structures
                method = str(self.method).lower()
                if method == "string" and len(structures) != 2:
                    raise ValueError("String requires exactly two endpoint structures.")
                if any(
                    getattr(getattr(atoms, "calc", None), "solvent_correction", None)
                    is not None
                    for atoms in structures
                ):
                    if method not in {"neb", "string"}:
                        raise ValueError(
                            "Implicit PRFO and Dimer require exactly one Atoms geometry; "
                            "multi-image TS admits only standard NEB or path-only String."
                        )
                    if method == "neb":
                        from .algorithm.neb import NEBParams
                        validate_path_parameters(
                            "neb",
                            {"neb": {key: value for key, value in self.params.items() if key != "method"}},
                            NEBParams,
                        )
                    else:
                        from .algorithm.string import GSMParams
                        validate_path_parameters(
                            "string",
                            {"string": {key: value for key, value in self.params.items() if key != "method"}},
                            GSMParams,
                        )
                    implicit_ts = bind_admitted_implicit_path(structures, method)
                    working = []
                    for image in structures:
                        copied = image.copy()
                        copied.calc = image.calc
                        working.append(copied)
                    self.atoms = working
                    path_structures = working
            if implicit_ts:
                method = str(self.method).lower()
                calculator = self.atoms.calc if isinstance(self.atoms, Atoms) else path_structures[0].calc
                if method in {"neb", "string"}:
                    derivative = "complete composed energy and forces only"
                elif method == "prfo":
                    derivative = (
                        "analytic composed Hessian"
                        if str(getattr(calculator, "hessian", "")).lower()
                        == "analytic"
                        else "numerical Hessian"
                    )
                else:
                    dimer_params = JobABC._select_subdict(
                        self.params, ("dimer", "DIMER", "ts")
                    )
                    derivative = (
                        "direct analytic composed HVP"
                        if dimer_params.get("use_hvp") is True
                        else "finite-difference curvature from the complete composed forces"
                    )
                self.log_info(
                    [
                        f"\nEXPERIMENTAL IMPLICIT-SOLVENT {self.method.upper()} BOUNDARY\n",
                        (
                            "This workflow uses the complete composed "
                            "MLIP+implicit-solvent potential with "
                            f"{derivative}. "
                            + (
                                "It is a path-generation capability, not a claim of "
                                "a converged MEP, transition state, or accuracy certification.\n"
                                if method in {"neb", "string"}
                                else
                                "It is a TS candidate-search capability, not a claim of "
                                "convergence or accuracy certification.\n"
                            )
                        ),
                    ]
                )

            cache_images = (
                [self.atoms]
                if isinstance(self.atoms, Atoms)
                else path_images(self.atoms)
            )
            with preserve_path_calculator_caches(cache_images, enabled=implicit_ts):
                return self._dispatch_method()

    def _dispatch_method(self):
        if self.method == 'prfo':
            from .algorithm import PRFO
            prfo = PRFO(
                atoms=cast(Atoms, self.atoms),
                output=self.output,
                paras=self.params,
            )
            return prfo.run()

        elif self.method == 'neb':
            structures = path_images(self.atoms)
            if len(structures) < 2:
                raise ValueError('NEB requires at least two structures.')
            from .algorithm import NEB
            neb = NEB(
                output=self.output,
                atoms_or_molecules=Molecules(structures),
                paras=self.params,
            )
            return neb.run()

        elif self.method == 'string':
            structures = path_images(self.atoms)
            if len(structures) != 2:
                raise ValueError('String requires exactly two endpoint structures.')
            from .algorithm import GSM
            string = GSM(
                output=self.output,
                atoms_R=structures[0],
                atoms_P=structures[1],
                paras=self.params
            )
            return string.run()

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
            return dimer.run()

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

from ase import Atoms
import numpy as np

from ..jobABC import JobABC

from maple.function.timer import timer


def _preselect_pure_nonmd_internal_mode(atoms):
    """Select the sole uncertainty-resolved internal negative mode once."""

    from ..pure_nonmd_status import is_pure_nonmd_v2
    if not is_pure_nonmd_v2(atoms):
        return None, None
    import numpy as np
    from maple.solvation.derivatives.molecular_modes import analyze_hessian_evaluation

    evaluation = atoms.calc.get_hessian_evaluation(atoms)
    analysis = analyze_hessian_evaluation(atoms, evaluation)
    if not analysis.is_resolved_index_one:
        raise InvalidStartingHessian(
            "IRC requires exactly one uncertainty-resolved negative internal mode."
        )
    mode_index = analysis.statuses.index("negative")
    mode_cart = np.asarray(analysis.modes_cartesian[mode_index], dtype=np.float64)
    root_mass = np.sqrt(np.repeat(atoms.get_masses(), 3))
    mode_mw = root_mass * mode_cart
    mode_mw /= np.linalg.norm(mode_mw)
    diagnostic = {
        "mode_index": mode_index,
        "mode_statuses": list(analysis.statuses),
        "resolved_negative_count": analysis.resolved_negative_count,
        "resolved_positive_count": analysis.resolved_positive_count,
        "uncertain_count": analysis.uncertain_count,
        "uncertainty_eV_per_A2_amu": analysis.uncertainty_eV_per_A2_amu,
        "hessian_evaluation_sha256": evaluation.evaluation_sha256,
    }
    return mode_mw, diagnostic


def _irc_status(method, atoms, result, mode_diagnostic, f_max_th, f_rms_th):
    from ..pure_nonmd_status import is_pure_nonmd_v2, make_status
    if not is_pure_nonmd_v2(atoms):
        return result
    branches, traces = {}, []
    for name in ("forward", "backward"):
        side = result[name]
        records = side["records"]
        last = records[-1] if records else None
        metrics = {} if last is None else {
            "energy_hartree": float(last["E"]),
            "max_force_hartree_per_angstrom": float(last["maxG"]),
            "rms_force_hartree_per_angstrom": float(last["rmsG"]),
        }
        branches[name] = {
            key: side[key] for key in (
                "converged", "termination_reason", "termination_class", "iterations"
            )
        }
        branches[name].update(
            final_metrics=metrics, endpoint_minimum_verified=False,
            geometry=None if last is None else {
                "positions_angstrom": np.asarray(last["x"], dtype=float).tolist()
            },
        )
        for key in ("error", "error_type"):
            if key in side:
                branches[name][key] = side[key]
        traces.append({"direction": name, "records": [
            {"energy_hartree": float(record["E"]),
             "max_force_hartree_per_angstrom": float(record["maxG"]),
             "rms_force_hartree_per_angstrom": float(record["rmsG"]),
             "positions_angstrom": np.asarray(record["x"], dtype=float).tolist()}
            for record in records
        ]})
    overall = all(branch["converged"] for branch in branches.values())
    classes = {branch["termination_class"] for branch in branches.values()}
    classification = next((value for value in
        ("execution_failure", "validation_failure", "bounded_nonconvergence")
        if value in classes), "converged")
    return make_status(
        workflow="irc", method=method, converged=overall,
        termination_reason="both_directions_converged" if overall else "one_or_more_directions_failed",
        termination_class=classification,
        iterations=sum(branch["iterations"] for branch in branches.values()),
        final_metrics={"forward": branches["forward"]["final_metrics"],
                       "backward": branches["backward"]["final_metrics"],
                       "mode": mode_diagnostic},
        atoms=atoms, trace=traces, directions=branches,
    )

class InvalidStartingHessian(RuntimeError):
    """The requested IRC seed is not a resolved internal index-one saddle."""


class IRC(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='gs'):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params

    def run(self):
        with timer("IRC Calculation"):
            try:
                mode, diagnostic = _preselect_pure_nonmd_internal_mode(self.atoms)
            except InvalidStartingHessian as exc:
                from ..pure_nonmd_status import make_status
                return make_status(
                    workflow="irc", method=self.method, converged=False, executed=False,
                    termination_class="validation_failure",
                    termination_reason="invalid_starting_hessian_index", iterations=0,
                    final_metrics={"error": str(exc)}, atoms=self.atoms,
                )
            if self.method == 'gs':
                from .algorithm import GS
                irc = GS(self.atoms, output=self.output, paras=self.commandcontrol)
                irc.preselected_mode_mw = mode
                irc.preselected_mode_diagnostic = diagnostic
                result = irc.run()
            elif self.method == 'hpc':
                from .algorithm import HPC
                irc = HPC(self.atoms, output=self.output, paras=self.commandcontrol)
                irc.preselected_mode_mw = mode
                irc.preselected_mode_diagnostic = diagnostic
                result = irc.run()
            elif self.method == 'eulerpc':
                from .algorithm import EulerPC
                irc = EulerPC(self.atoms, output=self.output, paras=self.commandcontrol)
                irc.preselected_mode_mw = mode
                irc.preselected_mode_diagnostic = diagnostic
                result = irc.run()
            elif self.method == 'lqa':
                from .algorithm import LQA
                irc = LQA(self.atoms, output=self.output, paras=self.commandcontrol)
                irc.preselected_mode_mw = mode
                irc.preselected_mode_diagnostic = diagnostic
                result = irc.run()
            else:
                raise NotImplementedError(f'IRC method {self.method} not implemented yet.')
            if mode is not None:
                return _irc_status(
                    self.method, self.atoms, result, diagnostic,
                    irc.p.f_max_th, irc.p.f_rms_th,
                )
            return result

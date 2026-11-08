from ase import Atoms
import numpy as np

from ..jobABC import JobABC

from maple.function.timer import timer


def _preselect_pure_nonmd_internal_mode(atoms, *, mode_index=None):
    """Select a certified V2 mode or a computed/exploratory V4 mode."""

    from ..pure_nonmd_status import is_pure_nonmd_v2, is_pure_torch_v4
    pure_v2 = is_pure_nonmd_v2(atoms)
    pure_v4 = is_pure_torch_v4(atoms)
    if not (pure_v2 or pure_v4):
        return None, None
    import numpy as np
    from maple.solvation.derivatives.molecular_modes import analyze_hessian_evaluation

    identified = (
        atoms.calc.get_identified_hessian_evaluation(atoms)
        if pure_v4 and mode_index is None else None
    )
    evaluation = (
        identified.evaluation if identified is not None
        else atoms.calc.get_hessian_evaluation(atoms)
    )
    combined = None
    if identified is not None:
        from maple.solvation.derivatives.computed_stationary import (
            analyze_identified_hessian,
        )

        combined = analyze_identified_hessian(atoms, identified)
        analysis = combined.mode_analysis
    else:
        analysis = analyze_hessian_evaluation(atoms, evaluation)
    if pure_v2 and not analysis.is_resolved_index_one:
        raise InvalidStartingHessian(
            "IRC requires exactly one uncertainty-resolved negative internal mode."
        )
    if pure_v2:
        selected_index = analysis.statuses.index("negative")
        selection_kind = "uncertainty_resolved_internal_negative"
    else:
        if mode_index is None:
            from maple.solvation.derivatives.computed_stationary import (
                computed_assessment_record,
            )

            computed = combined.stationary_assessment
            if not computed.computed_index_one:
                raise InvalidStartingHessian(
                    "Automatic V4 IRC requires a QA-passing computed index-one seed."
                )
            selected_index = int(
                np.flatnonzero(
                    computed.computed_eigenvalues_eV_per_A2_amu
                    < -computed.computed_resolution_radius
                )[0]
            )
            selection_kind = "computed_index_one_automatic"
            computed_record = computed_assessment_record(computed)
        else:
            if type(mode_index) is not int or mode_index < 0:
                raise InvalidStartingHessian(
                    "Pure Torch V4 IRC mode_index must be a non-negative integer."
                )
            if mode_index >= analysis.internal_dimension:
                raise InvalidStartingHessian("IRC mode_index is outside the internal spectrum.")
            eigenvalue = float(analysis.eigenvalues_eV_per_A2_amu[mode_index])
            if not np.isfinite(eigenvalue) or eigenvalue >= 0.0:
                raise InvalidStartingHessian(
                    "Pure Torch V4 mode_index must select a finite raw-negative internal mode."
                )
            selected_index = mode_index
            selection_kind = "explicit_raw_negative_candidate"
            computed_record = None
    mode_cart = np.asarray(analysis.modes_cartesian[selected_index], dtype=np.float64)
    root_mass = np.sqrt(np.repeat(atoms.get_masses(), 3))
    mode_mw = root_mass * mode_cart
    mode_mw /= np.linalg.norm(mode_mw)
    diagnostic = {
        "mode_index": selected_index,
        "mode_statuses": list(analysis.statuses),
        "resolved_negative_count": analysis.resolved_negative_count,
        "resolved_positive_count": analysis.resolved_positive_count,
        "uncertain_count": analysis.uncertain_count,
        "uncertainty_eV_per_A2_amu": analysis.uncertainty_eV_per_A2_amu,
        "hessian_evaluation_sha256": evaluation.evaluation_sha256,
    }
    if pure_v4:
        diagnostic.update({
            "selection_kind": selection_kind,
            "selected_raw_eigenvalue_eV_per_A2_amu": float(
                analysis.eigenvalues_eV_per_A2_amu[selected_index]
            ),
            "raw_internal_eigenvalues_eV_per_A2_amu": [
                float(value) for value in analysis.eigenvalues_eV_per_A2_amu
            ],
            "independent_hessian_uncertainty": None,
        })
        if computed_record is not None:
            diagnostic["computed_stationary_assessment"] = computed_record
    return mode_mw, diagnostic


def _endpoint_computed_assessment(atoms, positions):
    from maple.solvation.derivatives.computed_stationary import (
        analyze_identified_hessian,
        computed_assessment_record,
    )

    original = atoms.get_positions().copy()
    try:
        atoms.set_positions(np.asarray(positions, dtype=np.float64))
        identified = atoms.calc.get_identified_hessian_evaluation(atoms)
        return computed_assessment_record(
            analyze_identified_hessian(atoms, identified).stationary_assessment
        )
    finally:
        atoms.set_positions(original)


def _irc_status(method, atoms, result, mode_diagnostic, f_max_th, f_rms_th):
    from ..pure_nonmd_status import (
        is_pure_torch_v4, is_route2_workflow_status_enabled, make_status,
    )
    if not is_route2_workflow_status_enabled(atoms):
        return result
    branches, traces = {}, []
    for name in ("forward", "backward"):
        side = result[name]
        records = side["records"]
        last = records[-1] if records else None
        metrics: dict[str, object] = {} if last is None else {
            "energy_hartree": float(last["E"]),
            "max_force_hartree_per_angstrom": float(last["maxG"]),
            "rms_force_hartree_per_angstrom": float(last["rmsG"]),
        }
        if last is not None and "terminal_localization" in last:
            metrics["terminal_localization"] = dict(last["terminal_localization"])
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
        if (
            is_pure_torch_v4(atoms)
            and mode_diagnostic.get("selection_kind")
            == "computed_index_one_automatic"
            and last is not None
        ):
            try:
                endpoint = _endpoint_computed_assessment(atoms, last["x"])
                branches[name]["endpoint_computed_assessment"] = endpoint
                branches[name]["endpoint_computed_minimum"] = bool(
                    endpoint["computed_minimum"]
                )
            except Exception as exc:
                branches[name]["endpoint_computed_assessment_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                branches[name]["endpoint_computed_minimum"] = None
        for key in ("error", "error_type"):
            if key in side:
                branches[name][key] = side[key]
        trace_records = []
        for record in records:
            item = {
                "energy_hartree": float(record["E"]),
                "max_force_hartree_per_angstrom": float(record["maxG"]),
                "rms_force_hartree_per_angstrom": float(record["rmsG"]),
                "positions_angstrom": np.asarray(
                    record["x"], dtype=float
                ).tolist(),
            }
            if "terminal_localization" in record:
                item["terminal_localization"] = dict(
                    record["terminal_localization"]
                )
            trace_records.append(item)
        traces.append({"direction": name, "records": trace_records})
    overall = all(branch["converged"] for branch in branches.values())
    classes = {branch["termination_class"] for branch in branches.values()}
    classification = next((value for value in
        ("execution_failure", "validation_failure", "bounded_nonconvergence")
        if value in classes), "converged")
    auto_computed = (
        mode_diagnostic.get("selection_kind") == "computed_index_one_automatic"
    )
    endpoint_minima = {
        name: branches[name].get("endpoint_computed_minimum")
        for name in ("forward", "backward")
    }
    endpoint_validation_passed = (
        all(value is True for value in endpoint_minima.values())
        if auto_computed else None
    )
    task_converged = bool(
        overall and (endpoint_validation_passed is True if auto_computed else True)
    )
    if auto_computed and overall and not endpoint_validation_passed:
        classification = "validation_failure"
    return make_status(
        workflow="irc", method=method, converged=task_converged,
        termination_reason=(
            "both_directions_and_endpoint_minima_validated"
            if task_converged and auto_computed
            else "endpoint_computed_minimum_validation_failed"
            if auto_computed and overall
            else "both_directions_converged"
            if overall else "one_or_more_directions_failed"
        ),
        termination_class=classification,
        iterations=sum(branch["iterations"] for branch in branches.values()),
        final_metrics={
            "forward": branches["forward"]["final_metrics"],
            "backward": branches["backward"]["final_metrics"],
            "mode": mode_diagnostic,
            "numerical_stationary_point_validated": (
                task_converged if auto_computed else None
            ),
            "endpoint_computed_minima": endpoint_minima,
            "reaction_connectivity_verified": None,
        },
        atoms=atoms, trace=traces, directions=branches,
    )

class InvalidStartingHessian(RuntimeError):
    """The requested IRC seed is not a resolved internal index-one saddle."""


class IRC(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='gs'):
        super().__init__(output)
        from ..legacy_units import require_legacy_hartree_workflow_view
        require_legacy_hartree_workflow_view(atoms)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params

    def run(self):
        with timer("IRC Calculation"):
            try:
                mode_index = self.commandcontrol.get("mode_index")
                if mode_index is None:
                    mode, diagnostic = _preselect_pure_nonmd_internal_mode(self.atoms)
                else:
                    mode, diagnostic = _preselect_pure_nonmd_internal_mode(
                        self.atoms, mode_index=mode_index
                    )
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
                status = _irc_status(
                    self.method, self.atoms, result, diagnostic,
                    irc.p.f_max_th, irc.p.f_rms_th,
                )
                receipt = self.commandcontrol.get("_route2_irc_threshold_receipt")
                if receipt is not None:
                    status["final_metrics"]["effective_convergence_thresholds"] = dict(receipt)
                return status
            return result

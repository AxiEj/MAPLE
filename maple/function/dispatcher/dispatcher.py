from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, List, Protocol, Union
import math
import numpy as np

from ase import Atoms
from ..utility import Molecules


class ParamsCommandControl(Protocol):
    params: dict[str, Any]


@dataclass(slots=True)
class _MappingCommandControl:
    params: dict[str, Any]


def _normalize_command_control(
    control: ParamsCommandControl | Mapping[str, Any],
) -> ParamsCommandControl:
    if isinstance(control, MutableMapping):
        params = control if isinstance(control, dict) else dict(control)
        return _MappingCommandControl(params)
    if isinstance(control, Mapping):
        return _MappingCommandControl(dict(control))
    params = getattr(control, "params", None)
    if not isinstance(params, MutableMapping):
        raise TypeError(
            "commandcontrol must be a mapping or expose mutable mapping params."
        )
    normalized = params if isinstance(params, dict) else dict(params)
    return _MappingCommandControl(normalized)


def _apply_v4_irc_threshold_precedence(
    params, *, explicit, requested_level, method
):
    def positive_float(value, *, name):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a finite positive number.")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be a finite positive number.")
        return result

    resolved = {}
    for key in ("f_max_th", "f_rms_th"):
        source_value = explicit[key] if key in explicit else params[key]
        resolved[key] = positive_float(source_value, name=key)
    level_names = {
        "extratight", "tight", "medium", "loose", "extraloose", "superloose"
    }
    receipt = {
        "policy": "v4_explicit_over_level",
        "requested_level": requested_level,
        "endpoint_localization_candidate_cap": (
            12 if method in {"hpc", "eulerpc"} else None
        ),
    }
    for key in ("f_max_th", "f_rms_th"):
        if key in explicit:
            params[key] = resolved[key]
            source = "explicit"
        else:
            params[key] = resolved[key]
            source = (
                f"level:{requested_level}"
                if requested_level in level_names else "fallback:medium"
            )
        receipt[
            "f_max_hartree_per_angstrom"
            if key == "f_max_th" else "f_rms_hartree_per_angstrom"
        ] = float(params[key])
        receipt["f_max_source" if key == "f_max_th" else "f_rms_source"] = source
    params["_route2_irc_threshold_receipt"] = receipt
    return receipt

class Dispatcher():
    def __init__(self):
        pass

    def __call__(self, commandcontrol: ParamsCommandControl | Mapping[str, Any], jobtype: str, atoms: Union[Atoms, Molecules, List[Atoms]], output:str, extra:dict | None=None) -> object:
        from .legacy_units import legacy_hartree_job_calculators
        from .pure_nonmd_status import (
            REQUIRED_STATUS_FIELDS, NonMDWorkflowFailure, WorkflowOutcome,
            calculator_provenance, is_pure_nonmd_v2, is_pure_torch_calculator,
            is_pure_torch_v4, make_status, write_status,
        )
        from ..route2_smd_profiles import (
            normalize_frequency_method_request,
            route2_workflow_capability_for_params,
        )

        command = _normalize_command_control(commandcontrol)
        params = command.params
        images = atoms.multiatoms if isinstance(atoms, Molecules) else (
            atoms if isinstance(atoms, list) else [atoms]
        )
        v2_flags = [is_pure_nonmd_v2(image) for image in images]
        pure_v2 = bool(v2_flags) and all(v2_flags)
        torch_flags = [is_pure_torch_calculator(image) for image in images]
        pure_torch = bool(torch_flags) and all(torch_flags)
        v4_flags = [is_pure_torch_v4(image) for image in images]
        pure_torch_v4 = bool(v4_flags) and all(v4_flags)
        if any(v2_flags) and not pure_v2:
            raise ValueError(
                "pure non-MD v2 jobs require every image to use a canonical "
                "pure non-MD calculator."
            )
        if any(torch_flags) and not pure_torch:
            raise ValueError(
                "pure Torch jobs require every image to use the canonical "
                "analytic Torch calculator."
            )
        if jobtype == "freq" and "deprecated_alias" in params:
            raise ValueError(
                "deprecated_alias is derived internally and cannot be supplied."
            )
        frequency_request = (
            normalize_frequency_method_request(
                params.get("method"), pure_torch_v4=pure_torch_v4
            )
            if jobtype == "freq" else None
        )
        requested_method = str(
            params.get("ensemble") if jobtype == "md"
            else (
                frequency_request.canonical_method
                if frequency_request is not None
                else params.get("method") or jobtype
            )
        )
        capability = None
        sp_request = None
        if pure_torch_v4:
            raw_calc = getattr(images[0].calc, "raw_calculator", images[0].calc)
            if jobtype == "sp":
                from .sp.result import normalize_v4_sp_request

                sp_request = normalize_v4_sp_request(atoms, params)
            capability = route2_workflow_capability_for_params(
                raw_calc.profile_spec, str(jobtype), params
            )
            if capability is None or capability.execution_class == "closed":
                raise ValueError("Pure Torch V4 workflow capability is closed.")
            if len(images) != 1 and capability.image_policy != "multi":
                raise ValueError("Pure Torch V4 task requires one molecular structure.")
        elif pure_torch:
            if len(images) != 1:
                raise ValueError("pure Torch v3 does not support path or multi-image jobs.")
            if jobtype not in {"sp", "opt", "freq"}:
                raise ValueError(
                    "pure Torch v3 supports only SP, OPT, and FREQ; "
                    "TS, IRC, MD, and path workflows remain closed."
                )
            if (
                jobtype == "freq"
                and str(
                    params.get("method", "mw")
                ).lower()
                != "mw"
            ):
                raise ValueError("pure Torch v3 frequency supports only method=mw.")
        if pure_v2 or pure_torch_v4:
            configurations = set()
            physical_identities = {
                (
                    tuple(image.get_chemical_symbols()),
                    tuple(float(value) for value in image.get_masses()),
                    image.info.get("charge"), image.info.get("mult"),
                )
                for image in images
            }
            if len(physical_identities) != 1:
                raise ValueError(
                    "Route 2 workflow images must share symbols, masses, charge, and multiplicity."
                )
            for image in images:
                calculator = getattr(image.calc, "raw_calculator", image.calc)
                # Validate every image before the first algorithm/energy call.
                # Element order, electronic metadata and periodicity are part
                # of the calculator's existing domain contract.
                validation = calculator._validate_atoms(image)
                configurations.add(validation[1])
            if len(configurations) != 1:
                raise ValueError("Route 2 workflow images must share one PES configuration.")
            identities = {
                tuple(calculator_provenance(image).values()) for image in images
            }
            if len(identities) != 1:
                raise ValueError(
                    "Route 2 workflow images must share profile, scalar, and device."
                )
            constrained = any(
                tuple(getattr(image, "constraints", ()) or ()) for image in images
            )
            if (
                pure_torch_v4 and constrained
                and capability is not None
                and not capability.constraints_allowed
            ):
                raise ValueError(
                    "Pure Torch V4 Hessian/HVP workflows do not support atom constraints."
                )
        if pure_v2:
            if jobtype == "md":
                raise ValueError("MD is not supported by pure non-MD v2 profiles.")
            if jobtype not in {"sp", "opt", "scan", "freq", "ts", "irc"}:
                raise ValueError(f"Unsupported pure non-MD v2 task: {jobtype}")
            if (jobtype == "scan"
                    and str(params.get("mode", "relaxed")).lower() == "relaxed"
                    and str(params.get("method", "lbfgs")).lower() == "rfo"):
                raise ValueError(
                    "relaxed RFO scan is unsupported because a constrained "
                    "Hessian projection is not implemented."
                )
            if (jobtype == "freq"
                    and str(params.get("method", "mw")).lower() != "mw"):
                raise ValueError(
                    "pure non-MD v2 frequency supports only mass-weighted method=mw."
                )

        try:
            with legacy_hartree_job_calculators(atoms):
                result = self._dispatch_legacy(
                    command, jobtype, atoms, output, extra
                )
                if sp_request is not None:
                    from .sp.result import SPBatchResult, SPFrameResult

                    expected = SPBatchResult if sp_request.returns_batch else SPFrameResult
                    if not isinstance(result, expected):
                        raise RuntimeError(
                            "V4 SP returned an unexpected result shape; expected "
                            f"{expected.__name__}."
                        )
        except Exception as exc:
            if pure_v2 or pure_torch_v4:
                status = dict(exc.status) if isinstance(exc, NonMDWorkflowFailure) else make_status(
                    workflow=str(jobtype), method=requested_method,
                    converged=False, termination_reason="exception", iterations=0,
                    final_metrics={"error_type": type(exc).__name__, "error": str(exc)},
                    atoms=atoms, executed=False,
                    termination_class="execution_failure",
                    **calculator_provenance(images[0]),
                )
                status.update(calculator_provenance(images[0]))
                if pure_torch_v4:
                    status = WorkflowOutcome.from_internal_status(
                        status, capability=capability,
                    ).to_route2_record(capability.outcome_policy)
                try:
                    write_status(output, status)
                except Exception:
                    # Never replace the actual workflow exception with a
                    # secondary serialization/filesystem failure.
                    pass
            raise

        if pure_v2 or pure_torch_v4:
            attached_status = getattr(result, "_maple_nonmd_status", None)
            if attached_status is not None:
                result = attached_status
            if not (isinstance(result, dict) and all(
                    field in result for field in REQUIRED_STATUS_FIELDS)):
                final_metrics = {}
                workflow_converged = True
                workflow_termination_reason = "calculation_completed"
                workflow_termination_class = None
                sp_geometry = None
                raw_calc = getattr(images[0], "calc", None)
                raw_calc = getattr(raw_calc, "raw_calculator", raw_calc)
                if jobtype == "sp":
                    from .sp.result import SPBatchResult, SPFrameResult

                    if isinstance(result, SPFrameResult):
                        final_metrics["energy_eV"] = result.energy_eV
                        final_metrics["forces_eV_per_A"] = (
                            None
                            if result.forces_eV_per_A is None
                            else [list(row) for row in result.forces_eV_per_A]
                        )
                        final_metrics["force_scope"] = result.force_scope
                        if result.forces_eV_per_A is not None:
                            forces = np.asarray(
                                result.forces_eV_per_A, dtype=np.float64
                            )
                            final_metrics["max_abs_force_eV_per_A"] = float(
                                np.max(np.abs(forces))
                            )
                        sp_geometry = [{
                            "symbols": list(result.symbols),
                            "positions_angstrom": [
                                list(row) for row in result.positions_angstrom
                            ],
                        }]
                    elif isinstance(result, SPBatchResult):
                        final_metrics["frame_count"] = len(result.frames)
                        final_metrics["frames"] = result.to_records()
                        sp_geometry = [
                            {
                                "symbols": list(frame.symbols),
                                "positions_angstrom": [
                                    list(row) for row in frame.positions_angstrom
                                ],
                            }
                            for frame in result.frames
                        ]
                    elif not pure_torch_v4:
                        raw_results = getattr(raw_calc, "results", {}) or {}
                        if "energy" in raw_results:
                            final_metrics["energy_eV"] = float(raw_results["energy"])
                        if "forces" in raw_results:
                            forces = np.asarray(
                                raw_results["forces"], dtype=np.float64
                            )
                            final_metrics["forces_eV_per_A"] = forces.tolist()
                            final_metrics["max_abs_force_eV_per_A"] = float(
                                np.max(np.abs(forces))
                            )
                elif jobtype == "freq":
                    from .frequency.quantities import FrequencyQuantityResult

                    if isinstance(result, FrequencyQuantityResult):
                        final_metrics.update(result.to_final_metrics())
                        if not (
                            result.cartesian_curvature.computed_operator_quality_passed
                        ):
                            workflow_converged = False
                            workflow_termination_reason = "computed_operator_qa_failed"
                            workflow_termination_class = "validation_failure"
                    else:
                        analysis = getattr(raw_calc, "last_frequency_mode_analysis", None)
                        if analysis is not None:
                            final_metrics.update({
                                "rigid_rank": int(analysis.rigid_rank),
                                "internal_mode_count": int(analysis.internal_dimension),
                                "resolved_negative_count": int(analysis.resolved_negative_count),
                                "resolved_positive_count": int(analysis.resolved_positive_count),
                                "uncertain_count": int(analysis.uncertain_count),
                                "eigenvalues_eV_per_A2_amu": [
                                    float(value) for value in analysis.eigenvalues_eV_per_A2_amu
                                ],
                            })
                        evaluation = getattr(raw_calc, "last_hessian_evaluation", None)
                        if evaluation is not None:
                            final_metrics["hessian_eV_per_A2"] = (
                                np.asarray(
                                    evaluation.hessian_eV_per_A2,
                                    dtype=np.float64,
                                ).tolist()
                            )
                result = make_status(
                    workflow=str(jobtype), method=requested_method,
                    converged=workflow_converged,
                    termination_reason=workflow_termination_reason,
                    termination_class=workflow_termination_class,
                    iterations=0, final_metrics=final_metrics, atoms=atoms,
                )
                if sp_geometry is not None:
                    result["geometry"] = sp_geometry
            result.update(calculator_provenance(images[0]))
            if pure_torch_v4:
                result = WorkflowOutcome.from_internal_status(
                    result, capability=capability,
                ).to_route2_record(capability.outcome_policy)
            write_status(output, result)
            succeeded = result["converged"] if pure_v2 else result["task_success"]
            if not succeeded:
                raise RuntimeError(
                    f"Route 2 workflow {jobtype} did not succeed: "
                    f"{result['termination_reason']}; diagnostics retained in status sidecar."
                )
        return result

    def _dispatch_legacy(self, commandcontrol: ParamsCommandControl, jobtype: str, atoms: Union[Atoms, Molecules, List[Atoms]], output:str, extra:dict | None=None) -> object:

        """
        Dispatches the job based on the job type.
        Args:
            commandcontrol: CommandControl object
            jobtype: The type of job to be performed.
            atoms: The ASE Atoms object, Molecules object, or a list of Atoms objects.
            output: The path to the output file.
            extra: Extra parameters to be passed to the job.
        """

        self.output = output
        self.commandcontrol = commandcontrol
        params = commandcontrol.params
        from .pure_nonmd_status import is_pure_torch_v4
        v4_irc = bool(jobtype == "irc" and is_pure_torch_v4(atoms))
        explicit_irc = {
            key: params[key] for key in ("f_max_th", "f_rms_th") if key in params
        } if v4_irc else {}
        requested_level = (
            str(params["level"]).lower()
            if params.get("level") not in (None, "") else None
        )
        self.set_throshould(atoms)
        if v4_irc:
            _apply_v4_irc_threshold_precedence(
                params,
                explicit=explicit_irc,
                requested_level=requested_level,
                method=str(params.get("method") or "gs").lower(),
            )
            targets = atoms.multiatoms if isinstance(atoms, Molecules) else (
                atoms if isinstance(atoms, list) else [atoms]
            )
            for target in targets:
                setattr(target, "f_max_th", float(params["f_max_th"]))
                setattr(target, "f_rms_th", float(params["f_rms_th"]))
        if jobtype == 'opt':
            from .optimization import Optimization

            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For optimization job, only one Atoms object is allowed.')
            opt = Optimization(output=output, atoms=atoms, params=params)
            return opt.run()
            
        elif jobtype == 'sp':
            from .sp import SinglePoint

            sp_params = params

            # Handle trajectory/multiple structures
            if isinstance(atoms, Molecules):
                atoms_input = atoms.multiatoms
                sp = SinglePoint(output=output, atoms=atoms_input, paras=sp_params)
                return sp.run()
            elif isinstance(atoms, list):
                sp = SinglePoint(output=output, atoms=atoms, paras=sp_params)
                return sp.run()
            else:
                # Single structure (backward compatibility)
                sp = SinglePoint(output=output, atoms=atoms, paras=sp_params)
                return sp.run()

        elif jobtype == 'scan':
            from .scan import Scan

            if extra is not None:
                if 'scan' not in extra:
                    raise ValueError('Constraints not provided for scan job')
            else:
                raise ValueError('Constraints not provided for scan job')
            
            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For scan job, only one Atoms object is allowed.')

            method: Any = params.get("method")
            scan = Scan(output=output, atoms=atoms, method=method, constraints=extra['scan'], params=params)
            return scan.run()
            
        elif jobtype == 'freq':
            from .frequency import Frequency
            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For frequency job, only one Atoms object is allowed.')
            freq = Frequency(output=output, atoms=atoms, paras=params)
            return freq.run()
            
        elif jobtype == 'ts':
            from .ts import TransitionState
            
            # TS job allows Molecules object for methods like NEB, STRING
            if isinstance(atoms, (list, Molecules)):
                method = params.get('method')
                if method in ['neb', 'string', 'autoneb']:
                    # Convert Molecules to its internal list if needed
                    atoms_input = atoms.multiatoms if isinstance(atoms, Molecules) else atoms
                    ts = TransitionState(output=output, atoms=atoms_input, method=method, params=params)
                    return ts.run()
                elif method == 'prfo':
                    raise NotImplementedError('For transition state search job with PRFO method, only one Atoms object is allowed.')
                else:
                    raise ValueError(f'Unknown TS method: {method}')

            # Single Atoms object
            method: Any = params.get("method")
            ts = TransitionState(output=output, atoms=atoms, method=method, params=params)
            return ts.run()
        
        elif jobtype == 'irc':
            from .irc import IRC

            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For IRC job, only one Atoms object is allowed.')
            method: Any = params.get("method")
            irc = IRC(output=output, atoms=atoms, method=method, params=params)
            return irc.run()

        elif jobtype == 'md':
            from .md.ensemble.nve import NVE
            from .md.ensemble.nvt import NVT
            from .md.ensemble.npt import NPT

            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For MD job, only one Atoms object is allowed.')

            if 'charge' not in atoms.info or 'mult' not in atoms.info:
                raise ValueError(
                    "MD requires explicit charge and multiplicity. "
                    "Provide either 'XYZ <charge> <mult> <path>' or an inline 'charge mult' line before coordinates."
                )

            ensemble = params.get('ensemble', 'nve').lower()
            if ensemble == 'nve':
                md = NVE(output=output, atoms=atoms, paras=params)
            elif ensemble == 'nvt':
                md = NVT(output=output, atoms=atoms, paras=params)
            elif ensemble == 'npt':
                md = NPT(output=output, atoms=atoms, paras=params)
            else:
                raise ValueError(f"Unknown MD ensemble: '{ensemble}'")
            return md.run()
            
            
        else:
            try:
                raise NotImplementedError('Job type not implemented')
            except NotImplementedError as e:
                self.log_error(str(e))
                
    def set_throshould(self, atoms) -> None:
        """
        Sets the convergence throshould for the atoms object.

        Args:
            atoms: The ASE Atoms object, Molecules object, or list of Atoms.
        """
        
        # Geometry optimization convergence thresholds (Eh/Å and Å), Gaussian-style
        if self.commandcontrol.params.get('level') == 'extratight':
            self.commandcontrol.params['f_max_th'] = 0.00030
            self.commandcontrol.params['f_rms_th'] = 0.00020
            self.commandcontrol.params['dp_max_th'] = 0.00030
            self.commandcontrol.params['dp_rms_th'] = 0.00020

        elif self.commandcontrol.params.get('level') == 'tight':
            self.commandcontrol.params['f_max_th'] = 0.00085
            self.commandcontrol.params['f_rms_th'] = 0.00055
            self.commandcontrol.params['dp_max_th'] = 0.00110
            self.commandcontrol.params['dp_rms_th'] = 0.00075

        elif self.commandcontrol.params.get('level') == 'medium':
            self.commandcontrol.params['f_max_th'] = 0.00285
            self.commandcontrol.params['f_rms_th'] = 0.00190
            self.commandcontrol.params['dp_max_th'] = 0.00315
            self.commandcontrol.params['dp_rms_th'] = 0.00210

        elif self.commandcontrol.params.get('level') == 'loose':
            self.commandcontrol.params['f_max_th'] = 0.00380
            self.commandcontrol.params['f_rms_th'] = 0.00250
            self.commandcontrol.params['dp_max_th'] = 0.00600
            self.commandcontrol.params['dp_rms_th'] = 0.00400

        elif self.commandcontrol.params.get('level') == 'extraloose':
            self.commandcontrol.params['f_max_th'] = 0.00755
            self.commandcontrol.params['f_rms_th'] = 0.00500
            self.commandcontrol.params['dp_max_th'] = 0.01200
            self.commandcontrol.params['dp_rms_th'] = 0.00800

        elif self.commandcontrol.params.get('level') == 'superloose':
            self.commandcontrol.params['f_max_th'] = 0.08500
            self.commandcontrol.params['f_rms_th'] = 0.05500
            self.commandcontrol.params['dp_max_th'] = 0.14500
            self.commandcontrol.params['dp_rms_th'] = 0.09500

        else:
            # fallback to medium
            self.commandcontrol.params['f_max_th'] = 0.00285
            self.commandcontrol.params['f_rms_th'] = 0.00190
            self.commandcontrol.params['dp_max_th'] = 0.00315
            self.commandcontrol.params['dp_rms_th'] = 0.00210

        # Apply thresholds to atoms
        if isinstance(atoms, Molecules):
            # Apply to all atoms in Molecules object
            for atom in atoms.multiatoms:
                atom.f_max_th = self.commandcontrol.params['f_max_th']
                atom.f_rms_th = self.commandcontrol.params['f_rms_th']
                atom.dp_max_th = self.commandcontrol.params['dp_max_th']   
                atom.dp_rms_th = self.commandcontrol.params['dp_rms_th']
        elif isinstance(atoms, list):
            # Apply to all atoms in list
            for atom in atoms:
                atom.f_max_th = self.commandcontrol.params['f_max_th']
                atom.f_rms_th = self.commandcontrol.params['f_rms_th']
                atom.dp_max_th = self.commandcontrol.params['dp_max_th']   
                atom.dp_rms_th = self.commandcontrol.params['dp_rms_th']
        else:
            # Single Atoms object
            atoms.f_max_th = self.commandcontrol.params['f_max_th']
            atoms.f_rms_th = self.commandcontrol.params['f_rms_th']
            atoms.dp_max_th = self.commandcontrol.params['dp_max_th']   
            atoms.dp_rms_th = self.commandcontrol.params['dp_rms_th']



    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")

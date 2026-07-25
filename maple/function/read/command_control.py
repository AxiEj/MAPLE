import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional


class CommandControl:
    """
    Parse and validate input settings.
    One task only: sp/opt/ts/scan/freq/irc/md.
    All other settings are global parameters.
    """

    SUPPORTED_TASKS = {"sp", "opt", "ts", "scan", "freq", "irc", "md"}

    SUPPORTED_UMA_TASKS = {"omol", "omat", "oc20", "odac", "omc", "oc22", "oc25"}
    SUPPORTED_UMA_SIZES = {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}
    SUPPORTED_UMA_INFERENCE = {"default", "turbo"}
    UMA_DEFAULT_SIZE = "uma-s-1p1"  # keep in sync with _uma_calculator.UMA_DEFAULT_SIZE
    SUPPORTED_HESSIAN_MODES = {"analytic", "numerical"}

    DEFAULTS = {
        "model": None,
        "device": None,
        "d4": False,
        "sp": {},
        "opt": {},
        "ts": {},
        "irc": {"method": "gs"},
        "scan": {},
        "freq": {
            "method": "mw",
            "temperature": 298.15,
            "pressure_kpa": 101.325,
            "ilowfreq": 2,
            "verbosity": 1,
            "treat_imag_as_real": False,
            "device": "cpu",
        },
        "md": {
            "ensemble": "nve",
            "timestep": 0.25,
            "steps": 400000,
            "temperature": 300.0,
            "traj_every": 100,
            "log_every": 100,
            "init_velocities": True,
            "restart": False,
            "load_state": False,
            "rst_file": "",
            "rst_every": 1000,
            "remove_com": True,
            "remove_com_every": 100,
            "remove_rotation": False,
            "remove_angular": False,
            "remove_angular_every": 0,
            "random_seed": None,
            "thermostat": "langevin",
            "friction": 0.001,
            "tau_t": 100.0,
            "barostat": "c-rescale",
            "pressure": 1.0,
            "tau_p": 2000.0,
            "compressibility": 4.5e-5,
            "mdp": None,
            "traj_format": "xyz",
            "debug": False,
        },
    }

    IMPLEMENTATION_MAP = {
        "opt": {"lbfgs", "rfo", "sd", "cg", "sdcg", ""},
        "scan": {"lbfgs", "rfo", "sd", "cg", "sdcg"},
        "ts": {"prfo", "string", "neb", "dimer", "autoneb"},
        "freq": {"mw", "nonmw"},
        "sp": set(),
        "irc": {"gs", "hpc", "eulerpc", "lqa"},
        "md": {"nve", "nvt", "npt"},
    }
    GLOBAL_PARAMS = {
        "model",
        "model_options",
        "device",
        "gpuid",
        "d4",
        "pbc",
        "solv",
        "charge",
        "level",
    }
    LBFGS_PARAMS = {
        "memory",
        "curvature",
        "max_step",
        "max_iter",
        "verbose",
        "log_final_paths",
    }
    RFO_PARAMS = {
        "max_iter",
        "trust_radius_init",
        "trust_radius_min",
        "trust_radius_max",
        "eta_shrink",
        "eta_expand",
        "evals_eps",
        "mu_margin",
        "max_bisect_it",
        "verbose",
        "log_final_paths",
    }
    SDCG_PARAMS = {
        "max_step",
        "max_iter",
        "verbose",
        "sd_enabled",
        "cg_enabled",
        "sd_max_iter",
        "cg_switch_fmax",
        "cg_restart_threshold",
        "cg_beta_method",
        "diis_enabled",
        "diis_store_every",
        "diis_min_snapshots",
        "diis_memory",
        "log_final_paths",
    }
    OPT_METHOD_PARAMS = {
        "lbfgs": LBFGS_PARAMS,
        "rfo": RFO_PARAMS,
        "sd": SDCG_PARAMS,
        "cg": SDCG_PARAMS,
        "sdcg": SDCG_PARAMS,
    }
    SCAN_PARAMS = {"method", "mode"}
    SOLV_PARAMS = {
        "method",
        "implicit",
        "explicit",
        "inner",
        "solvent",
        "radius",
        "padding",
        "shape",
        "box_size",
        "density",
        "density_scale",
        "number",
        "clash_method",
        "tolerance",
        "vdw_scale",
        "vdw_fallback_radius",
        "seed",
        "randomize",
        "experimental",
        "provider",
        "profile",
        "model",
        "nonpolar",
        "platform",
        "executable",
        "grid_spacing",
        "grid_points",
        "probe_radius",
        "surface_tension",
        "pressure",
        "timeout",
        "write_shell",
        "shell_cutoff",
        "solvent_pdb",
        # Compatibility aliases / explicit rejections.
        "clash_cutoff",
        "write_cell",
    }
    CHARGE_PARAMS = {
        "source",
        "method",
        "mode",
        "label",
        "geometry",
        "executable",
        "timeout",
    }
    SOLV_REMOVED_PARAMS = {
        "fix_dis": (
            "Explicit solvent 'fix_dis' has been removed: clusters are now "
            "non-periodic and no atom constraints are written. Use "
            "write_shell=true with shell_cutoff=<Å> to export a solute-centred "
            "shell instead."
        ),
    }

    VALIDATED_TASK_PARAMS = {"opt", "scan", "freq", "md"}

    TS_REFINE_MAP = {
        "neb": {"cineb", "nebts"},
        "string": {"cistring", "stringts"},
    }

    def __init__(
        self, params: Dict[str, Any], task: str, output_path: Optional[str] = None
    ):
        self.params = params
        self.task = task
        self.output_path = output_path

    @classmethod
    def from_settings(
        cls,
        settings_lines: List[str],
        output_path: Optional[str] = None,
    ) -> "CommandControl":
        params: Dict[str, Any] = {}
        task: Optional[str] = None
        seen_keys = set()
        log_lines = ["Parsing # commands...\n"]

        for raw in settings_lines:
            line = raw.strip()
            if not line.startswith("#"):
                continue

            match = re.match(
                r"#\s*([A-Za-z0-9_]+)\s*(?:=\s*([^()\s]+))?\s*(?:\((.*)\))?",
                line,
            )
            if not match:
                continue

            key = match.group(1).strip().lower()
            assign_val = match.group(2)
            paren_val = match.group(3)

            if key in cls.SUPPORTED_TASKS:
                if task and task != key:
                    cls._log_error(
                        output_path, f"Multiple tasks defined: '{task}' and '{key}'."
                    )
                    raise ValueError(f"Multiple tasks defined: '{task}' and '{key}'.")

                task = key
                params.update(cls.DEFAULTS.get(key, {}))
                log_lines.append(f"Task set to '{task}'\n")

                inline_md_keys = set()
                if paren_val:
                    cls._parse_nested(params, paren_val)
                    if task == "md":
                        inline_md_keys = {
                            kv.split("=", 1)[0].strip().lower()
                            for kv in paren_val.split(",")
                            if "=" in kv
                        }

                if task == "md" and params.get("mdp"):
                    cls._load_mdp(params, inline_md_keys, output_path)

                continue

            if key in seen_keys:
                cls._log_error(output_path, f"Duplicate parameter: '{key}'.")
                raise ValueError(f"Duplicate parameter: '{key}'.")
            seen_keys.add(key)

            if paren_val is not None and assign_val is not None:
                sub = {}
                cls._parse_nested(sub, paren_val)
                params[key] = cls._auto_cast(assign_val.strip())
                params[f"{key}_options"] = sub
                log_lines.append(
                    f"Global parameter: {key} = {params[key]} with options {sub}\n"
                )
                continue

            if paren_val:
                if key == "pbc":
                    params["pbc"] = cls._parse_pbc(paren_val, output_path)
                    log_lines.append(f"Global parameter: pbc = {params['pbc']}\n")
                    continue

                sub = {}
                cls._parse_nested(sub, paren_val)
                params[key] = sub
                log_lines.append(f"Global nested parameter: {key} = {sub}\n")
                continue

            if assign_val:
                value = cls._auto_cast(assign_val.strip())
                params[key] = value
                log_lines.append(f"Global parameter: {key} = {value}\n")
                continue

            params[key] = True
            log_lines.append(f"Global flag: {key} = True\n")

        if not task:
            task = "sp"
            params.update(cls.DEFAULTS.get("sp", {}))
            log_lines.append("No task specified. Defaulting to 'sp'.\n")

        cls._normalize_params(params)
        cls._normalize_method_flags(params, task, output_path)
        cls._validate(params, task, output_path)
        cls._log_info(output_path, log_lines)

        return cls(params, task, output_path)

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().replace("\ufeff", "").lower()

    @staticmethod
    def _parse_nested(target: Dict[str, Any], inner: str) -> None:
        for kv in inner.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                target[CommandControl._normalize_key(k)] = CommandControl._auto_cast(
                    v.strip()
                )
            else:
                target[CommandControl._normalize_key(kv)] = True

    @classmethod
    def _parse_pbc(cls, inner: str, output_path: Optional[str]) -> List[float]:
        try:
            values = [float(x.strip()) for x in inner.lstrip("=").strip().split(",")]
        except ValueError as exc:
            cls._log_error(output_path, f"Invalid PBC values: {inner} - {exc}")
            raise ValueError(f"Invalid PBC values: {inner}") from exc

        if len(values) == 2:
            a, b = values
            cellpar = [a, b, 1000.0, 90.0, 90.0, 90.0]
        elif len(values) == 3:
            a, b, c = values
            cellpar = [a, b, c, 90.0, 90.0, 90.0]
        elif len(values) == 6:
            cellpar = values
        else:
            cls._log_error(
                output_path, f"PBC requires 2, 3, or 6 values, got {len(values)}."
            )
            raise ValueError(
                f"PBC requires 2, 3, or 6 values (a,b[,c][,alpha,beta,gamma]), got {len(values)}."
            )

        return cellpar

    @staticmethod
    def _auto_cast(value: str) -> Any:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            pass
        return value

    @classmethod
    def _load_mdp(
        cls,
        params: Dict[str, Any],
        inline_keys: set[str],
        output_path: Optional[str] = None,
    ) -> None:
        from ..dispatcher.md.mdp_reader import parse_mdp

        mdp_path = params["mdp"]
        try:
            mdp_params = parse_mdp(mdp_path)
        except FileNotFoundError:
            cls._log_error(output_path, f"MDP file not found: {mdp_path!r}")
            raise
        except ValueError as exc:
            cls._log_error(output_path, str(exc))
            raise

        defaults = cls.DEFAULTS.get("md", {})
        for key, mdp_val in mdp_params.items():
            if key in defaults and key not in inline_keys:
                params[key] = mdp_val

        if "remove_rotation" in mdp_params and "remove_angular" not in mdp_params:
            params["remove_angular"] = params["remove_rotation"]

    @classmethod
    def _normalize_params(cls, params: Dict[str, Any]) -> None:
        if "remove_angular" not in params and "remove_rotation" in params:
            params["remove_angular"] = params["remove_rotation"]
        if params.get("remove_angular"):
            params["remove_com"] = True

        if "model" in params and params["model"] is not None:
            params["model"] = str(params["model"]).strip().lower()

        model_options = params.get("model_options")
        if isinstance(model_options, dict):
            for key in ("task", "size", "hessian", "inference"):
                if key in model_options and isinstance(model_options[key], str):
                    model_options[key] = model_options[key].lower()

        solv_options = params.get("solv")
        if isinstance(solv_options, dict):
            for key in (
                "shape",
                "explicit",
                "method",
                "implicit",
                "solvent",
                "clash_method",
                "provider",
                "profile",
                "model",
                "nonpolar",
                "platform",
            ):
                if key in solv_options and isinstance(solv_options[key], str):
                    solv_options[key] = solv_options[key].lower()
            if solv_options.get("shape") == "box":
                solv_options["shape"] = "cube"

        charge_options = params.get("charge")
        if isinstance(charge_options, dict):
            for key in ("source", "method", "mode", "geometry"):
                if key in charge_options and isinstance(charge_options[key], str):
                    charge_options[key] = charge_options[key].lower()

        if "ensemble" in params and isinstance(params["ensemble"], str):
            params["ensemble"] = params["ensemble"].lower()

    @classmethod
    def _normalize_method_flags(
        cls, params: Dict[str, Any], task: str, output_path: Optional[str]
    ) -> None:
        allowed = {
            method for method in cls.IMPLEMENTATION_MAP.get(task, set()) if method
        }
        flags = sorted(method for method in allowed if params.get(method) is True)
        if not flags:
            return
        if len(flags) > 1:
            msg = f"Multiple method flags for task '{task}': {flags}"
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        method = flags[0]
        current = params.get("method")
        if current is not None and current != method:
            msg = (
                f"Conflicting method settings for task '{task}': "
                f"'{current}' and '{method}'"
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        params["method"] = method
        del params[method]

    @classmethod
    def _raise_unknown_param(
        cls,
        output_path: Optional[str],
        context: str,
        key: str,
        allowed: set[str],
    ) -> None:
        msg = f"Unknown {context} parameter: '{key}'."
        match = get_close_matches(key, sorted(allowed), n=1, cutoff=0.72)
        if match:
            msg += f" Did you mean '{match[0]}'?"
        cls._log_error(output_path, msg)
        raise ValueError(msg)

    @classmethod
    def _allowed_task_params(
        cls, task: str, params: Dict[str, Any]
    ) -> Optional[set[str]]:
        if task not in cls.VALIDATED_TASK_PARAMS:
            return None

        allowed = set(cls.GLOBAL_PARAMS)
        if task == "md":
            allowed.update(cls.DEFAULTS["md"])
            return allowed
        if task == "freq":
            allowed.update(cls.DEFAULTS["freq"])
            return allowed

        method = str(params.get("method") or "lbfgs").lower()
        method_params = cls.OPT_METHOD_PARAMS.get(method)
        if method_params is None:
            method_params = set().union(*cls.OPT_METHOD_PARAMS.values())

        allowed.add("method")
        allowed.update(method_params)
        if task == "scan":
            allowed.update(cls.SCAN_PARAMS)
        return allowed

    @classmethod
    def _validate_unknown_params(
        cls, params: Dict[str, Any], task: str, output_path: Optional[str]
    ) -> None:
        allowed = cls._allowed_task_params(task, params)
        if allowed is not None:
            context = task.upper()
            for key in params:
                if key not in allowed:
                    cls._raise_unknown_param(output_path, context, key, allowed)

        if "solv" in params:
            solv_params = params["solv"]
            if not isinstance(solv_params, dict):
                msg = "Solvation settings must use '#solv(key=value,...)' syntax."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            for key in solv_params:
                if key in cls.SOLV_REMOVED_PARAMS:
                    msg = cls.SOLV_REMOVED_PARAMS[key]
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                if key not in cls.SOLV_PARAMS:
                    cls._raise_unknown_param(
                        output_path, "solvation", key, cls.SOLV_PARAMS
                    )
        if "charge" in params:
            charge_params = params["charge"]
            if not isinstance(charge_params, dict):
                msg = "Charge settings must use '#charge(key=value,...)' syntax."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            for key in charge_params:
                if key not in cls.CHARGE_PARAMS:
                    cls._raise_unknown_param(
                        output_path, "charge", key, cls.CHARGE_PARAMS
                    )

    @classmethod
    def _validate_charge(
        cls, params: Dict[str, Any], *, required: bool, output_path: Optional[str]
    ) -> None:
        charge = params.get("charge")
        if charge is None:
            if required:
                msg = "Implicit solvation requires an explicit #charge(...) selection."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            return
        if not isinstance(charge, dict):
            return
        source = str(charge.get("source", "")).lower()
        if source not in {"mol2", "maple"}:
            msg = "#charge source must be 'mol2' or 'maple'."
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        method = charge.get("method")
        mode = str(charge.get("mode", "fixed")).lower()
        geometry = str(charge.get("geometry", "keep")).lower()
        if mode not in {"fixed", "polarizable"}:
            msg = "#charge mode must be 'fixed' or 'polarizable'."
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        if geometry not in {"keep", "provider"}:
            msg = "#charge geometry must be 'keep' or 'provider'."
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        if source == "mol2":
            if method is not None:
                msg = "#charge(source=mol2) reads fixed charges and does not accept method=."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if mode != "fixed":
                msg = "#charge(source=mol2) supports mode=fixed only."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if geometry != "keep":
                msg = "#charge(source=mol2) does not run a geometry provider; use geometry=keep."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            ignored = sorted(key for key in ("executable", "timeout") if key in charge)
            if ignored:
                msg = (
                    "#charge(source=mol2) does not run a charge executable; remove "
                    + ", ".join(ignored)
                    + "."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
        else:
            if method is None:
                method = "am1bcc"
            method = str(method).lower()
            if method not in {"am1bcc", "abcg2", "qeq-gto"}:
                msg = "MAPLE charge method must be am1bcc, abcg2, or qeq-gto."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            charge["method"] = method
            if mode == "polarizable" and method != "qeq-gto":
                msg = "mode=polarizable is available only for method=qeq-gto."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if method == "qeq-gto" and geometry != "keep":
                msg = "QEq-GTO does not optimize geometry; use geometry=keep."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if method == "qeq-gto":
                ignored = sorted(
                    key for key in ("executable", "timeout") if key in charge
                )
                if ignored:
                    msg = (
                        "QEq-GTO is native and does not use " + ", ".join(ignored) + "."
                    )
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
            if "label" in charge:
                msg = "#charge label is only valid for source=mol2 fixed-charge provenance."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
        if "timeout" in charge and (
            isinstance(charge["timeout"], bool)
            or not isinstance(charge["timeout"], (int, float))
            or charge["timeout"] <= 0
        ):
            msg = "#charge timeout must be a positive number of seconds."
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        for key in ("label", "executable"):
            if key in charge and (
                not isinstance(charge[key], str) or not charge[key].strip()
            ):
                msg = f"#charge {key} must be a non-empty string."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
        charge["source"] = source
        charge["mode"] = mode
        charge["geometry"] = geometry

    @classmethod
    def _validate_solvation(
        cls, params: Dict[str, Any], task: str, output_path: Optional[str]
    ) -> None:
        solv_params = params.get("solv")
        if not isinstance(solv_params, dict):
            return

        solvent_alias = solv_params.pop("solvent", None)
        if solvent_alias is not None:
            explicit = solv_params.get("explicit")
            implicit = solv_params.get("implicit")
            if explicit is None and implicit is None:
                # Backward-compatible interpretation:
                #   #solv(method=gbsa, solvent=water) -> implicit solvent
                #   #solv(solvent=water)              -> explicit solvent
                target = "implicit" if solv_params.get("method") else "explicit"
                solv_params[target] = solvent_alias
            elif solvent_alias not in {explicit, implicit}:
                msg = (
                    "Conflicting solvation alias 'solvent': use explicit=<name> "
                    "and/or implicit=<name> directly when they differ."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        for key in ("randomize", "write_shell", "experimental"):
            if key in solv_params and not isinstance(solv_params[key], bool):
                msg = f"Solvation {key} must be 'true' or 'false'."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        method = solv_params.get("method")
        explicit = solv_params.get("explicit")
        implicit = solv_params.get("implicit")
        inner = solv_params.get("inner")

        if inner is not None:
            if not isinstance(inner, str) or not inner.strip():
                msg = "Solvation inner must be the non-empty string 'prebuilt'."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            inner = inner.lower()
            if inner != "prebuilt":
                msg = (
                    "The explicit-inner/implicit-outer release supports "
                    "inner=prebuilt only."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            solv_params["inner"] = inner

        if method is not None:
            method = str(method).lower()
            solv_params["method"] = method
            if method == "gbsa":
                msg = (
                    "Legacy method=gbsa is no longer mapped silently. Migrate to "
                    "#solv(implicit=water,method=gb,model=obc2,nonpolar=ace) and add #charge(...)."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if method not in {"gb", "pb"}:
                msg = "Implicit solvation method must be 'gb' or 'pb'."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if explicit is not None and implicit is not None:
            msg = "Use either explicit=<solvent> or implicit=<solvent>, not both."
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        if inner is not None and implicit is None:
            msg = "inner=prebuilt requires implicit=<solvent>."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if implicit is not None:
            if method not in {"gb", "pb"}:
                msg = "Implicit solvation requires method=gb or method=pb."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            implicit = str(implicit).lower()
            if implicit != "water":
                msg = (
                    "The first implicit-solvation release supports implicit=water only."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            solv_params["implicit"] = implicit
            if solv_params.get("experimental") is not True:
                msg = (
                    "Implicit-solvation providers are not yet scientifically certified against "
                    "the public benchmark gate; add experimental=true to acknowledge this status."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "pbc" in params:
                msg = "Implicit solvation is non-periodic; remove #pbc."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            explicit_only = {
                "radius",
                "padding",
                "shape",
                "box_size",
                "density",
                "density_scale",
                "number",
                "clash_method",
                "tolerance",
                "vdw_scale",
                "vdw_fallback_radius",
                "seed",
                "randomize",
                "write_shell",
                "shell_cutoff",
                "solvent_pdb",
                "clash_cutoff",
                "write_cell",
            }
            conflicts = sorted(key for key in explicit_only if key in solv_params)
            if conflicts:
                msg = (
                    "Explicit-solvent options cannot be combined with implicit "
                    f"solvation: {', '.join(conflicts)}."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            cls._validate_charge(params, required=True, output_path=output_path)
            charge = params["charge"]
            if inner == "prebuilt" and charge.get("source") != "mol2":
                msg = (
                    "inner=prebuilt requires #charge(source=mol2) with fixed "
                    "per-atom charges for the complete cluster."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if inner == "prebuilt":
                prebuilt_provider = str(
                    solv_params.get(
                        "provider",
                        "openmm" if method == "gb" else "apbs",
                    )
                ).lower()
                prebuilt_nonpolar = str(
                    solv_params.get(
                        "nonpolar",
                        "ace" if method == "gb" else prebuilt_provider,
                    )
                ).lower()
                if (
                    method != "gb"
                    or prebuilt_provider != "openmm"
                    or prebuilt_nonpolar not in {"ace", "lcpo"}
                ):
                    msg = (
                        "inner=prebuilt is validated only with OpenMM GB and "
                        "nonpolar=ACE or LCPO."
                    )
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
            if method == "gb":
                provider = str(solv_params.get("provider", "openmm")).lower()
                if provider == "ambertools":
                    conflicts = sorted(
                        {
                            "platform",
                            "grid_spacing",
                            "grid_points",
                            "probe_radius",
                            "surface_tension",
                            "pressure",
                        }.intersection(solv_params)
                    )
                    if conflicts:
                        msg = (
                            "GB/AmberTools CHA-GB does not use APBS/OpenMM options: "
                            + ", ".join(conflicts)
                            + "."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    model = str(solv_params.get("model", "chagb")).lower()
                    profile = str(
                        solv_params.get("profile", "chagb-bondi-pbsa-inp2")
                    ).lower()
                    nonpolar = str(
                        solv_params.get("nonpolar", "cavity-dispersion")
                    ).lower()
                    if model != "chagb":
                        msg = "provider=ambertools requires model=chagb."
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    if profile != "chagb-bondi-pbsa-inp2":
                        msg = (
                            "provider=ambertools requires "
                            "profile=chagb-bondi-pbsa-inp2."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    if nonpolar != "cavity-dispersion":
                        msg = (
                            "provider=ambertools,model=chagb requires "
                            "nonpolar=cavity-dispersion."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    if (
                        charge.get("source") != "maple"
                        or charge.get("method") != "am1bcc"
                        or charge.get("mode") != "fixed"
                        or charge.get("geometry") != "keep"
                    ):
                        msg = (
                            "The validated CHA-GB profile requires fixed "
                            "#charge(source=maple,method=am1bcc,geometry=keep)."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    if inner is not None:
                        msg = (
                            "The CHA-GB SP profile does not support " "inner=prebuilt."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    if task != "sp" or params.get("verbose", 0) >= 1:
                        msg = (
                            "AmberTools CHA-GB/cavity-dispersion is "
                            "single-point energy-only."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                elif provider == "openmm":
                    pb_only = {
                        "executable",
                        "grid_spacing",
                        "grid_points",
                        "probe_radius",
                        "surface_tension",
                        "pressure",
                        "timeout",
                    }
                    conflicts = sorted(pb_only.intersection(solv_params))
                    if conflicts:
                        msg = (
                            "GB/OpenMM does not use PB provider options: "
                            + ", ".join(conflicts)
                            + "."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    model = str(solv_params.get("model", "obc2")).lower()
                    profiles = {
                        "hct": "hct-mbondi",
                        "obc1": "obc1-mbondi2",
                        "obc2": "obc2-mbondi2",
                        "gbn": "gbn-bondi",
                        "gbn2": "gbn2-mbondi3",
                    }
                    if model not in profiles:
                        msg = (
                            "GB/OpenMM model must be hct, obc1, obc2, " "gbn, or gbn2."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    profile = str(solv_params.get("profile", profiles[model])).lower()
                    if profile != profiles[model]:
                        msg = (
                            f"GB profile {profile!r} does not match "
                            f"model={model}; expected profile={profiles[model]}. "
                            "Arbitrary parameter mixing is disabled."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    nonpolar = str(solv_params.get("nonpolar", "ace")).lower()
                    if nonpolar not in {"ace", "lcpo", "none"}:
                        msg = (
                            "GB nonpolar must be ace or lcpo "
                            "(none is diagnostic only)."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    if (
                        nonpolar == "none"
                        and solv_params.get("experimental") is not True
                    ):
                        msg = (
                            "nonpolar=none is diagnostic only and requires "
                            "experimental=true."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                    if task == "md":
                        if charge.get("mode", "fixed") != "fixed":
                            msg = (
                                "Route 1 implicit-GB MD currently requires fixed charges; "
                                "polarizable research charge modes are not admitted."
                            )
                            cls._log_error(output_path, msg)
                            raise ValueError(msg)
                        if inner == "prebuilt":
                            msg = (
                                "inner=prebuilt MD is not admitted until fixed-shell "
                                "membership and occupancy constraints are defined."
                            )
                            cls._log_error(output_path, msg)
                            raise ValueError(msg)
                        ensemble = str(params.get("ensemble", "nve")).lower()
                        if ensemble not in {"nve", "nvt"}:
                            msg = (
                                "Implicit GB is non-periodic and supports only "
                                "non-periodic NVE or NVT MD; NPT is not available."
                            )
                            cls._log_error(output_path, msg)
                            raise ValueError(msg)
                    elif task == "freq":
                        if str(params.get("method", "mw")).lower() != "mw":
                            msg = (
                                "Route 1 implicit-GB frequency analysis supports "
                                "only method=mw; non-mass-weighted analysis is "
                                "not a physical thermochemistry path."
                            )
                            cls._log_error(output_path, msg)
                            raise ValueError(msg)
                        hessian_mode = (
                            params.get("model_options", {}).get("hessian")
                            if isinstance(params.get("model_options"), dict)
                            else None
                        )
                        if hessian_mode != "numerical":
                            msg = (
                                "Implicit GB frequency analysis requires explicit "
                                "#model=...(hessian=numerical) so the Hessian "
                                "differentiates the complete MLIP+GB force."
                            )
                            cls._log_error(output_path, msg)
                            raise ValueError(msg)
                        if charge.get("mode", "fixed") != "fixed":
                            msg = (
                                "Route 1 implicit-GB frequency analysis currently "
                                "requires fixed charges."
                            )
                            cls._log_error(output_path, msg)
                            raise ValueError(msg)
                    elif task not in {"sp", "opt", "scan"}:
                        msg = (
                            "GB currently supports SP, OPT, SCAN/PES, explicit "
                            "numerical FREQ, and fixed-charge non-periodic "
                            "NVE/NVT MD tasks only."
                        )
                        cls._log_error(output_path, msg)
                        raise ValueError(msg)
                else:
                    msg = "GB provider must be openmm or ambertools."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                solv_params.update(
                    model=model, provider=provider, profile=profile, nonpolar=nonpolar
                )
            elif method == "pb":
                if "platform" in solv_params:
                    msg = "PB/APBS does not use the OpenMM platform option."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                model = str(solv_params.get("model", "lpb")).lower()
                provider = str(solv_params.get("provider", "apbs")).lower()
                if model != "lpb":
                    msg = "PB model must be lpb in the first release."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                if provider not in {"apbs", "amber-pbsa"}:
                    msg = "PB provider must be apbs or amber-pbsa."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                expected_profile = (
                    "generic-mbondi2" if provider == "apbs" else "abcg2-pbsa-2023"
                )
                profile = str(solv_params.get("profile", expected_profile)).lower()
                if profile != expected_profile:
                    msg = f"provider={provider} requires profile={expected_profile}."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                nonpolar = str(
                    solv_params.get(
                        "nonpolar", "apbs" if provider == "apbs" else "amber-pbsa"
                    )
                ).lower()
                if nonpolar != provider:
                    msg = "PB polar and nonpolar terms must come from the same locked provider profile."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                if provider == "amber-pbsa" and charge.get("method") != "abcg2":
                    msg = "profile=abcg2-pbsa-2023 requires #charge(source=maple,method=abcg2)."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                if charge.get("mode") == "polarizable":
                    msg = "Polarizable QEq-PB is deferred; use mode=fixed for PB."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                if task != "sp" or params.get("verbose", 0) >= 1:
                    msg = "PB is single-point energy-only until force/grid convergence is certified."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                solv_params.update(
                    model=model, provider=provider, profile=profile, nonpolar=nonpolar
                )
            for key in (
                "grid_spacing",
                "probe_radius",
                "surface_tension",
                "pressure",
                "timeout",
            ):
                if key in solv_params and (
                    isinstance(solv_params[key], bool)
                    or not isinstance(solv_params[key], (int, float))
                ):
                    msg = f"Implicit-solvent {key} must be numeric."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
            for key in ("grid_spacing", "timeout"):
                if key in solv_params and solv_params[key] <= 0:
                    msg = f"Implicit-solvent {key} must be positive."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
            for key in ("probe_radius", "surface_tension", "pressure"):
                if key in solv_params and solv_params[key] < 0:
                    msg = f"Implicit-solvent {key} must be non-negative."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
            if (
                "grid_points" in solv_params
                and type(solv_params["grid_points"]) is not int
            ):
                msg = "Implicit-solvent grid_points must be an integer."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "grid_points" in solv_params and (
                solv_params["grid_points"] < 33
                or (solv_params["grid_points"] - 1) % 32 != 0
            ):
                msg = "APBS grid_points must have the nlev=4 form c*32+1 (65, 97, 129, ...)."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            for key in ("platform", "executable"):
                if key in solv_params and (
                    not isinstance(solv_params[key], str)
                    or not solv_params[key].strip()
                ):
                    msg = f"Implicit-solvent {key} must be a non-empty string."
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
            return

        if method is not None:
            msg = "method=gb/pb requires implicit=water; omit method for explicit solvent."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "experimental" in solv_params:
            msg = "experimental is only valid with an implicit PB/GB configuration."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if explicit is None:
            if solv_params:
                msg = "Solvation requires either explicit=<solvent> or implicit=<solvent>."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            return

        if "pbc" in params:
            msg = (
                "Explicit solvent clusters are non-periodic coordinate-only clusters; "
                "remove #pbc or use a periodic solvent backend."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "write_cell" in solv_params:
            msg = (
                "Explicit solvent clusters are non-periodic; "
                "write_cell/PBC output is not supported."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        shape = str(solv_params.get("shape", "sphere")).lower()
        if shape == "box":
            shape = "cube"
        solv_params["shape"] = shape
        if shape not in {"sphere", "cube"}:
            msg = "Explicit solvent shape must be 'sphere' or 'cube'."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "seed" in solv_params and type(solv_params["seed"]) is not int:
            msg = (
                "Explicit solvent seed must be an integer; "
                "use seed=-1 for non-reproducible sampling."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "number" in solv_params and (
            type(solv_params["number"]) is not int or solv_params["number"] < 0
        ):
            msg = "Explicit solvent number must be an integer >= 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        numeric_keys = (
            "radius",
            "padding",
            "box_size",
            "density",
            "density_scale",
            "tolerance",
            "vdw_scale",
            "vdw_fallback_radius",
            "shell_cutoff",
            "clash_cutoff",
        )
        for key in numeric_keys:
            if key in solv_params and (
                isinstance(solv_params[key], bool)
                or not isinstance(solv_params[key], (int, float))
            ):
                msg = f"Explicit solvent {key} must be numeric."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if "solvent_pdb" in solv_params:
            if (
                not isinstance(solv_params["solvent_pdb"], str)
                or not solv_params["solvent_pdb"].strip()
            ):
                msg = "Explicit solvent solvent_pdb must be a non-empty path string."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if "clash_method" not in solv_params:
            if "tolerance" in solv_params or "clash_cutoff" in solv_params:
                solv_params["clash_method"] = "distance"
            else:
                solv_params["clash_method"] = "vdw"

        if solv_params["clash_method"] not in {"vdw", "distance"}:
            msg = "Explicit solvent clash_method must be 'vdw' or 'distance'."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "padding" in solv_params and solv_params["padding"] <= 0:
            msg = "Explicit solvent padding must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if shape == "sphere":
            if "padding" in solv_params and "radius" in solv_params:
                msg = (
                    "Explicit solvent padding derives the sphere radius from the "
                    "solute envelope; do not combine padding with radius."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            radius = solv_params.get("radius", 10.0)
            if radius <= 0:
                msg = "Explicit solvent radius must be > 0 for shape=sphere."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "padding" not in solv_params:
                solv_params.setdefault("radius", radius)
            if "box_size" in solv_params:
                msg = "Explicit solvent box_size is only valid for shape=cube."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
        else:
            if "radius" in solv_params:
                msg = "Explicit solvent radius is only valid for shape=sphere."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "padding" in solv_params and "box_size" in solv_params:
                msg = (
                    "Explicit solvent padding derives the cube box_size from the "
                    "solute envelope; do not combine padding with box_size."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "padding" not in solv_params and "box_size" not in solv_params:
                msg = "Explicit solvent shape=cube requires box_size."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "box_size" in solv_params and solv_params["box_size"] <= 0:
                msg = "Explicit solvent box_size must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if "density" in solv_params and solv_params["density"] <= 0:
            msg = "Explicit solvent density must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "density_scale" in solv_params and solv_params["density_scale"] <= 0:
            msg = "Explicit solvent density_scale must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if solv_params["clash_method"] == "vdw":
            if "tolerance" in solv_params or "clash_cutoff" in solv_params:
                msg = (
                    "Explicit solvent tolerance/clash_cutoff are only valid "
                    "with clash_method=distance."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "vdw_scale" in solv_params and solv_params["vdw_scale"] <= 0:
                msg = "Explicit solvent vdw_scale must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if (
                "vdw_fallback_radius" in solv_params
                and solv_params["vdw_fallback_radius"] <= 0
            ):
                msg = "Explicit solvent vdw_fallback_radius must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
        elif "tolerance" in solv_params and solv_params["tolerance"] <= 0:
            msg = "Explicit solvent tolerance must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if solv_params["clash_method"] == "distance" and (
            "vdw_scale" in solv_params or "vdw_fallback_radius" in solv_params
        ):
            msg = (
                "Explicit solvent vdw_scale/vdw_fallback_radius are only valid "
                "with clash_method=vdw."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if (
            solv_params["clash_method"] == "distance"
            and "clash_cutoff" in solv_params
            and solv_params["clash_cutoff"] <= 0
        ):
            msg = "Explicit solvent clash_cutoff must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if solv_params.get("write_shell", False):
            if "shell_cutoff" not in solv_params:
                msg = "Explicit solvent write_shell=true requires shell_cutoff."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if solv_params["shell_cutoff"] <= 0:
                msg = "Explicit solvent shell_cutoff must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

    @classmethod
    def _validate(
        cls, params: Dict[str, Any], task: str, output_path: Optional[str]
    ) -> None:
        model = params.get("model")
        # Calculator names and class-declared model_options are registry-owned:
        # SetCalculator imports builtins, honors module= / MAPLE_CALCULATOR_PLUGINS,
        # and validates class OPTION_KEYS before construction.
        cls._validate_unknown_params(params, task, output_path)
        cls._validate_solvation(params, task, output_path)
        cls._validate_charge(params, required=False, output_path=output_path)

        if (
            "gpuid" in params
            and params["gpuid"] is not None
            and not isinstance(params["gpuid"], int)
        ):
            cls._log_error(output_path, "GPU ID must be an integer.")
            raise ValueError("GPU ID must be an integer.")

        if "d4" in params and not isinstance(params["d4"], bool):
            cls._log_error(output_path, "D4 must be 'true' or 'false'.")
            raise ValueError("D4 must be 'true' or 'false'.")

        if task == "sp":
            if "verbosity" in params:
                cls._log_error(output_path, "SP uses 'verbose', not 'verbosity'.")
                raise ValueError("SP uses 'verbose', not 'verbosity'.")
            if "verbose" in params and (
                type(params["verbose"]) is not int or params["verbose"] not in {0, 1}
            ):
                cls._log_error(output_path, "SP verbose must be 0 or 1.")
                raise ValueError("SP verbose must be 0 or 1.")

        if task == "freq":
            ilowfreq = params.get("ilowfreq")
            if type(ilowfreq) is not int or ilowfreq not in {0, 1, 2, 3}:
                msg = "FREQ ilowfreq must be one of 0, 1, 2, or 3."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            verbosity = params.get("verbosity")
            if type(verbosity) is not int or verbosity < 0:
                msg = "FREQ verbosity must be a non-negative integer."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if "method" in params:
            if task == "md":
                cls._log_error(
                    output_path,
                    "'method' is not valid for MD tasks; use 'ensemble=' instead.",
                )
                raise ValueError(
                    "'method' is not valid for MD tasks. Use 'ensemble=' to choose nve/nvt/npt."
                )

            allowed = cls.IMPLEMENTATION_MAP.get(task, set())
            if allowed and params["method"] not in allowed:
                cls._log_error(
                    output_path,
                    f"Method '{params['method']}' not implemented for task '{task}'.",
                )
                raise ValueError(
                    f"Method '{params['method']}' not implemented for task '{task}'."
                )

        if task == "ts" and "refine" in params:
            method = params.get("method")
            allowed_refines = (
                cls.TS_REFINE_MAP.get(method) if isinstance(method, str) else None
            )
            if allowed_refines is None:
                cls._log_error(
                    output_path, f"'refine' is not valid for TS method '{method}'."
                )
                raise ValueError(f"'refine' is not valid for TS method '{method}'.")
            if params["refine"] not in allowed_refines:
                cls._log_error(
                    output_path,
                    f"Refine '{params['refine']}' not implemented for TS method '{method}'.",
                )
                raise ValueError(
                    f"Refine '{params['refine']}' not implemented for TS method '{method}'."
                )

        if task == "md":
            ensemble = params.get("ensemble", "nve")
            allowed = cls.IMPLEMENTATION_MAP["md"]
            if ensemble not in allowed:
                cls._log_error(output_path, f"MD ensemble '{ensemble}' not supported.")
                raise ValueError(
                    f"MD ensemble '{ensemble}' not supported. Choose from: {sorted(allowed)}"
                )
        elif "ensemble" in params:
            cls._log_error(
                output_path, f"'ensemble' is only valid for MD tasks, not '{task}'."
            )
            raise ValueError(f"'ensemble' is only valid for MD tasks, not '{task}'.")

        if "pbc" in params:
            pbc_val = params["pbc"]
            if not isinstance(pbc_val, list) or len(pbc_val) != 6:
                cls._log_error(
                    output_path,
                    "PBC must be a list of 6 values [a, b, c, alpha, beta, gamma].",
                )
                raise ValueError("PBC must be a list of 6 values.")
            if any(pbc_val[i] <= 0 for i in range(3)):
                cls._log_error(
                    output_path, "PBC lattice parameters (a, b, c) must be positive."
                )
                raise ValueError("PBC lattice parameters must be positive.")
            if any(pbc_val[i] <= 0 or pbc_val[i] >= 180 for i in range(3, 6)):
                cls._log_error(
                    output_path,
                    "PBC angles (alpha, beta, gamma) must be in range (0, 180).",
                )
                raise ValueError("PBC angles must be in range (0, 180).")

        model_options = params.get("model_options", {})
        if model == "uma":
            task_opt = model_options.get("task")
            if task_opt is not None and task_opt not in cls.SUPPORTED_UMA_TASKS:
                msg = f"Unsupported UMA task: '{task_opt}'. Supported: {sorted(cls.SUPPORTED_UMA_TASKS)}"
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            size_opt = model_options.get("size")
            if size_opt is not None and size_opt not in cls.SUPPORTED_UMA_SIZES:
                msg = f"Unsupported UMA size: '{size_opt}'. Supported: {sorted(cls.SUPPORTED_UMA_SIZES)}"
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            inference_opt = model_options.get("inference")
            if (
                inference_opt is not None
                and inference_opt not in cls.SUPPORTED_UMA_INFERENCE
            ):
                msg = (
                    f"Unsupported UMA inference mode: '{inference_opt}'. "
                    f"Supported: {sorted(cls.SUPPORTED_UMA_INFERENCE)}"
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            if size_opt is None:
                # Make the actual checkpoint visible in summary() / .out.
                model_options.setdefault("size", cls.UMA_DEFAULT_SIZE)
                params["model_options"] = model_options

            if "pbc" in params and task_opt == "omol":
                cls._log_error(output_path, "PBC is incompatible with UMA task='omol'.")
                raise ValueError("PBC is incompatible with UMA task='omol'.")

        hessian_mode = model_options.get("hessian")
        if hessian_mode is not None and hessian_mode not in cls.SUPPORTED_HESSIAN_MODES:
            msg = (
                f"Unsupported Hessian mode: '{hessian_mode}'. "
                f"Supported: {sorted(cls.SUPPORTED_HESSIAN_MODES)}"
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

    @staticmethod
    def _log_info(output_path: Optional[str], lines: List[str]) -> None:
        if output_path:
            with open(output_path, "a") as handle:
                for line in lines:
                    handle.write(line)

    @staticmethod
    def _log_error(output_path: Optional[str], message: str) -> None:
        if output_path:
            with open(output_path, "a") as handle:
                handle.write(f"ERROR: {message}\n")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.params.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.params)
        out["task"] = self.task
        return out

    def summary(self) -> str:
        lines = ["Parsed configuration:\n", "-" * 40 + "\n"]
        lines.append(f"Task: {self.task}\n")
        model_options = self.params.get("model_options") or {}
        for key, value in self.params.items():
            if key == "model_options":
                continue  # folded into the model line below
            if key == "model" and isinstance(value, str) and model_options:
                opt_str = ",".join(f"{k}={v}" for k, v in model_options.items())
                value = f"{value}({opt_str})"
            lines.append(f"{key:<15}: {value}\n")
        return "".join(lines)

    def __repr__(self) -> str:
        return f"CommandControl(task={self.task}, params={self.params})"

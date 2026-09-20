
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Tuple


class JobABC(ABC):

    def __init__(self, output: str):
        self.output = output

    @abstractmethod
    def run(self):
        pass

    # ==============================================
    # Parameter handling utilities
    # ==============================================

    @staticmethod
    def _lower_keys(d: dict) -> dict:
        """Return a copy of dict with all string keys lowercased."""
        if not isinstance(d, dict):
            return {}
        return {(k.lower() if isinstance(k, str) else k): v for k, v in d.items()}

    @staticmethod
    def _select_subdict(paras: dict, name_aliases: Tuple[str, ...]) -> dict:
        """Extract a sub-dict using aliases (e.g., 'neb', 'NEB')."""
        if not isinstance(paras, dict):
            return {}
        low = JobABC._lower_keys(paras)
        for alias in name_aliases:
            key = alias.lower()
            if key in low and isinstance(low[key], dict):
                return low[key]
        return low

    @staticmethod
    def _update_dataclass_from_dict(dc_obj, d: dict):
        """Update a dataclass instance from a dict (case-insensitive keys)."""
        if not isinstance(d, dict):
            return dc_obj
        low = JobABC._lower_keys(d)
        fld_names = {f.name.lower(): f.name for f in fields(dc_obj)}
        for k_low, v in low.items():
            if k_low in fld_names:
                setattr(dc_obj, fld_names[k_low], v)
        return dc_obj

    def _init_params(self, params_class, paras: dict, aliases: Tuple[str, ...]):
        """Initialize params dataclass from external dict."""
        params = params_class()
        if isinstance(paras, dict):
            sub_dict = self._select_subdict(paras, aliases)
            self._update_dataclass_from_dict(params, sub_dict)
        return params

    # ==============================================
    # Logging utilities
    # ==============================================

    def log_inference_precision(self, calculator) -> None:
        """Expose backend-selected numerical precision without inventing defaults."""
        precision = getattr(calculator, "inference_precision_provenance", None)
        if not isinstance(precision, dict):
            return
        lines = [
            (
                f"Inference precision: requested={precision.get('requested_dtype', '<not reported>')}, "
                f"effective={precision.get('effective_dtype', '<not reported>')}\n"
            ),
        ]
        checkpoint_sha = precision.get("original_checkpoint_sha256")
        if checkpoint_sha is not None:
            lines.append(f"Original checkpoint SHA256: {checkpoint_sha}\n")
        self.log_info(lines)

    def log_numerical_hessian_diagnostics(self, calculator) -> None:
        """Report raw finite-difference quality, not symmetry imposed afterward."""
        diagnostics = getattr(calculator, "last_numerical_hessian_diagnostics", None)
        if not isinstance(diagnostics, dict):
            return
        self.log_info([
            "\nNumerical Hessian diagnostics (before symmetrization):\n",
            f"  Force-FD step: {diagnostics['cartesian_displacement_angstrom']:.8g} Angstrom\n",
            (
                "  Maximum raw antisymmetry: "
                f"{diagnostics['maximum_raw_asymmetry_hartree_per_angstrom2']:.8e} Hartree/Angstrom^2\n"
            ),
            (
                "  Relative raw antisymmetry (Frobenius): "
                f"{diagnostics['relative_raw_asymmetry_frobenius']:.8e}\n"
            ),
        ])
        self.log_inference_precision(calculator)

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

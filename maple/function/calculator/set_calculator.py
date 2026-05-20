import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import ase
import numpy as np
import torch
from ase import Atoms

from .ani._ani_calculator import ANICalculator
from .aimnet.options import (
    AIMNET_LEGACY_MODELS,
    AIMNET_PBC_MODELS,
    validate_aimnet_options,
)
from .mace._mace_calculator import MACECalculator
from .mace.options import (
    MACE_PBC_MODELS,
    MACEPOL_PBC_MODELS,
    validate_mace_pbc_options,
    validate_macepol_pbc_options,
)


IMPLEMENTATION_MODELS = [
    "ani2x",
    "ani1x",
    "ani1ccx",
    "ani1xnr",
    "maceoff23s",
    "maceoff23m",
    "maceoff23l",
    "egret",
    "aimnet2",
    "aimnet2nse",
    "aimnet2-pbc",
    "aimnet2nse-pbc",
    "mace-mp-pbc-small",
    "mace-mp-pbc-medium",
    "mace-mp-pbc-large",
    "macepol-pbc-small",
    "macepol-pbc-medium",
    "macepol-pbc-large",
    "uma",
    "maceomol",
    "macepols",
    "macepolm",
    "macepoll",
]

MODEL_NAME_TO_FILE = {
    "ani2x": "ani2x.pt",
    "ani1x": "ani1x.pt",
    "ani1ccx": "ani1ccx.pt",
    "ani1xnr": "ani1xnr.pt",
    "aimnet2": "aimnet2.pt",
    "aimnet2nse": "aimnet2nse.pt",
    "maceoff23m": "maceoff23m.pt",
    "maceomol": "maceomol.pt",
    "egret": "egret1s.pt",
    "uma-s-1p1": "uma-s-1p1.pt",
    "uma-s-1p2": "uma-s-1p2.pt",
    "uma-m-1p1": "uma-m-1p1.pt",
    "macepols": "macepols.pt",
    "macepolm": "macepolm.pt",
    "macepoll": "macepoll.pt",
}

HF_REPO_ID = "Wayne7815/MAPLE_model"

MODEL_HESSIAN_SUPPORT = {
    "ani2x": ("analytic", "numerical"),
    "ani1x": ("analytic", "numerical"),
    "ani1ccx": ("analytic", "numerical"),
    "ani1xnr": ("analytic", "numerical"),
    "maceoff23s": ("analytic", "numerical"),
    "maceoff23m": ("analytic", "numerical"),
    "maceoff23l": ("analytic", "numerical"),
    "egret": ("analytic", "numerical"),
    "aimnet2": ("analytic", "numerical"),
    "aimnet2nse": ("analytic", "numerical"),
    "aimnet2-pbc": (),
    "aimnet2nse-pbc": (),
    "mace-mp-pbc-small": (),
    "mace-mp-pbc-medium": (),
    "mace-mp-pbc-large": (),
    "macepol-pbc-small": (),
    "macepol-pbc-medium": (),
    "macepol-pbc-large": (),
    "uma": ("numerical",),
    "maceomol": ("analytic", "numerical"),
    "macepols": ("analytic", "numerical"),
    "macepolm": ("analytic", "numerical"),
    "macepoll": ("analytic", "numerical"),
}

MODEL_PBC_MD_SUPPORT = {
    "ani2x": False,
    "ani1x": False,
    "ani1ccx": False,
    "ani1xnr": False,
    "maceoff23s": False,
    "maceoff23m": False,
    "maceoff23l": False,
    "egret": False,
    "aimnet2": False,
    "aimnet2nse": False,
    "aimnet2-pbc": True,
    "aimnet2nse-pbc": True,
    "mace-mp-pbc-small": True,
    "mace-mp-pbc-medium": True,
    "mace-mp-pbc-large": True,
    "macepol-pbc-small": True,
    "macepol-pbc-medium": True,
    "macepol-pbc-large": True,
    "uma": True,
    "maceomol": False,
    "macepols": False,
    "macepolm": False,
    "macepoll": False,
}

MODEL_STRESS_SUPPORT = {
    "ani2x": False,
    "ani1x": False,
    "ani1ccx": False,
    "ani1xnr": False,
    "maceoff23s": False,
    "maceoff23m": False,
    "maceoff23l": False,
    "egret": False,
    "aimnet2": False,
    "aimnet2nse": False,
    "aimnet2-pbc": True,
    "aimnet2nse-pbc": True,
    "mace-mp-pbc-small": True,
    "mace-mp-pbc-medium": True,
    "mace-mp-pbc-large": True,
    "macepol-pbc-small": True,
    "macepol-pbc-medium": True,
    "macepol-pbc-large": True,
    "uma": True,
    "maceomol": False,
    "macepols": False,
    "macepolm": False,
    "macepoll": False,
}

UNSUPPORTED_CHARGE_MULT_MODELS = {
    "ani2x",
    "ani1x",
    "ani1ccx",
    "ani1xnr",
    "maceoff23s",
    "maceoff23m",
    "maceoff23l",
    "egret",
    "maceomol",
    "mace-mp-pbc-small",
    "mace-mp-pbc-medium",
    "mace-mp-pbc-large",
}


def model_supports_pbc_md(model: str) -> bool:
    """Return whether a MAPLE model has declared periodic MD support."""
    return bool(MODEL_PBC_MD_SUPPORT.get(model, False))


def model_supports_stress(model: str) -> bool:
    """Return whether a MAPLE model has declared calculator stress support."""
    return bool(MODEL_STRESS_SUPPORT.get(model, False))


def _calculator_neighbor_cutoff_A(calculator) -> Optional[float]:
    for attr in ("neighbor_cutoff_A", "cutoff_A", "maple_neighbor_cutoff"):
        cutoff = getattr(calculator, attr, None)
        if cutoff is None:
            continue
        try:
            return float(cutoff)
        except (TypeError, ValueError):
            return None
    return None


def _minimum_image_radius_A(atoms) -> tuple[float, float]:
    """Return the Wigner-Seitz inradius for the active periodic lattice.

    The minimum-image cutoff must be smaller than half the shortest non-zero
    periodic lattice vector, not merely half the shortest stored cell edge.  For
    skew triclinic cells, the shortest periodic image can be a linear
    combination such as ``a - b``.  ASE's Minkowski reduction gives a reduced
    basis whose first active vector is the lattice shortest vector, following
    Nguyen & Stehlé, ACM Trans. Algorithms 5, 46 (2009).
    """
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if not np.any(pbc):
        raise ValueError("minimum-image radius requires at least one periodic direction")

    cell = np.asarray(atoms.get_cell(), dtype=float)
    active_cell = cell[pbc]
    if not np.all(np.isfinite(active_cell)):
        raise ValueError("PBC cell vectors must be finite for minimum-image validation.")

    dim = int(np.sum(pbc))
    if np.linalg.matrix_rank(active_cell) < dim:
        raise ValueError(
            "PBC cell vectors must be linearly independent for minimum-image validation."
        )

    from ase.geometry import minkowski_reduce

    reduced_cell, _ = minkowski_reduce(cell, pbc=pbc)
    reduced_lengths = np.linalg.norm(np.asarray(reduced_cell, dtype=float)[pbc], axis=1)
    finite_lengths = reduced_lengths[np.isfinite(reduced_lengths) & (reduced_lengths > 0.0)]
    if finite_lengths.size == 0:
        raise ValueError("Could not determine a finite periodic lattice vector length.")

    shortest_lattice_vector = float(np.min(finite_lengths))
    return 0.5 * shortest_lattice_vector, shortest_lattice_vector


def _validate_pbc_neighbor_cutoff(atoms, calculator) -> None:
    """Validate cutoff against the true minimum-image radius.

    The cutoff bound follows Allen & Tildesley, Computer Simulation of Liquids,
    2nd ed. (Oxford University Press, 2017), section 1.5: each pair interaction
    should see at most one periodic image.  For skew cells this requires the
    Wigner-Seitz inradius, i.e. half the shortest non-zero periodic lattice
    vector, rather than half the shortest stored cell edge.
    """
    if atoms is None or not any(atoms.pbc):
        return
    if not getattr(calculator, "maple_pbc_md_supported", False):
        return

    cutoff = _calculator_neighbor_cutoff_A(calculator)
    if cutoff is None or not np.isfinite(cutoff) or cutoff <= 0.0:
        return

    radius, shortest_lattice_vector = _minimum_image_radius_A(atoms)
    if cutoff >= radius:
        raise ValueError(
            f"Backend neighbor cutoff {cutoff:.3f} A >= minimum-image radius "
            f"{radius:.3f} A (shortest periodic lattice vector = "
            f"{shortest_lattice_vector:.3f} A). "
            "Minimum-image convention is violated; enlarge the cell or reduce the cutoff."
        )


class SetClaculator:
    def __init__(
        self,
        device: torch.device,
        model: str,
        output: str,
        atoms: Optional[Atoms] = None,
        d4: bool = False,
        implicit: str = "None",
        solvent: str = "None",
        model_options: Optional[dict] = None,
    ) -> None:
        self.output = output
        self.model = model
        self.d4 = d4
        self.device = device
        self.atoms = atoms
        self.implicit = implicit
        self.solvent = solvent
        self.model_options = model_options or {}

    def _validate_requested_hessian_mode(self) -> None:
        mode = self.model_options.get("hessian")
        if mode is None:
            return

        mode = str(mode).lower()
        declared = MODEL_HESSIAN_SUPPORT.get(self.model)
        if declared is not None and mode not in declared:
            if not declared:
                raise ValueError(f"Model '{self.model}' does not support Hessian modes.")
            supported_text = ", ".join(sorted(declared))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f"Supported modes: {supported_text}"
            )

    def _apply_hessian_mode(self, calculator) -> None:
        mode = self.model_options.get("hessian")
        if mode is None:
            return

        mode = str(mode).lower()
        supported = getattr(calculator, "supported_hessian_modes", None)
        if supported is not None and mode not in supported:
            if not supported:
                raise ValueError(f"Model '{self.model}' does not support Hessian modes.")
            supported_text = ", ".join(sorted(supported))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f"Supported modes: {supported_text}"
            )

        if not hasattr(calculator, "hessian"):
            raise ValueError(f"Model '{self.model}' does not expose configurable Hessian modes.")

        calculator.hessian = mode

    def _validated_aimnet_options(self) -> dict:
        if self.model in AIMNET_LEGACY_MODELS:
            return validate_aimnet_options(self.model_options)
        if self.model in AIMNET_PBC_MODELS:
            return validate_aimnet_options(self.model_options, pbc=True)
        return {}

    def _validated_mace_pbc_options(self) -> dict:
        if self.model not in MACE_PBC_MODELS:
            return {}
        return validate_mace_pbc_options(self.model_options)

    def _validated_macepol_pbc_options(self) -> dict:
        if self.model not in MACEPOL_PBC_MODELS:
            return {}
        return validate_macepol_pbc_options(self.model_options)

    def _coerce_uma_inference_for_device(
        self, inference: Optional[str], device_name: str
    ) -> Optional[str]:
        if inference == "turbo" and device_name == "cpu":
            self.log_info([
                "\n [WARNING] UMA inference='turbo' requires CUDA; "
                "falling back to 'default' on CPU.\n"
            ])
            return "default"
        return inference

    def _ensure_model_file(self, model_name: str) -> Optional[Path]:
        filename = MODEL_NAME_TO_FILE.get(model_name)
        if filename is None:
            return None

        model_dir = Path(__file__).parent / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if model_path.exists():
            return model_path

        url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{filename}"
        self.log_info(
            [
                f" [INFO] Model file '{filename}' not found locally.\n",
                f" [INFO] Downloading from: {url}\n",
            ]
        )

        temp_path = model_path.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url) as response:
                size = response.headers.get("Content-Length")
                if size:
                    self.log_info([f" [INFO] File size: {int(size) / 1024 / 1024:.1f} MB\n"])
                with open(temp_path, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            temp_path.rename(model_path)
            self.log_info([f" [INFO] Download complete: {model_path}\n"])
        except urllib.error.HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            self.log_error(f" [ERROR] Download failed (HTTP {exc.code}): {url}\n")
            raise RuntimeError(f"Failed to download model '{model_name}': HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            temp_path.unlink(missing_ok=True)
            self.log_error(f" [ERROR] Network error: {exc.reason}\n")
            raise RuntimeError(f"Failed to download model '{model_name}': {exc.reason}") from exc

        return model_path

    def _warn_charge_mult(self) -> None:
        if self.atoms is None:
            return

        has_charge = self.atoms.info.get("charge", 0) != 0
        has_mult = self.atoms.info.get("mult", 1) != 1
        if (has_charge or has_mult) and self.model in UNSUPPORTED_CHARGE_MULT_MODELS:
            self.log_info(
                [
                    f"\n [WARNING] Model '{self.model}' does not support charge/multiplicity.\n",
                    f"           charge={self.atoms.info.get('charge', 0)}, ",
                    f"mult={self.atoms.info.get('mult', 1)} will be IGNORED.\n",
                    "           Models with charge/mult support: aimnet2, aimnet2nse, uma, macepols/m/l, macepol-pbc-small/medium/large\n",
                ]
            )

    def _annotate_calculator_capabilities(self, calculator) -> None:
        calculator.maple_model_name = self.model
        calculator.maple_model_options = dict(self.model_options)
        calculator.maple_pbc_md_supported = model_supports_pbc_md(self.model)
        calculator.maple_stress_supported = model_supports_stress(self.model)

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        model = self.model

        if model in {"ani2x", "ani1x", "ani1ccx", "ani1xnr"}:
            self._ensure_model_file(model)
            calculator = ANICalculator(
                model=model,
                d4=self.d4,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"maceoff23s", "maceoff23m", "maceoff23l", "egret"}:
            self._ensure_model_file(model)
            calculator = MACECalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in AIMNET_LEGACY_MODELS:
            aimnet_options = self._validated_aimnet_options()
            self._ensure_model_file(model)
            from .aimnet._aimnet2_calculator import AIMNet2Calculator

            calculator = AIMNet2Calculator(
                model=model,
                device=self.device,
                coulomb_method=aimnet_options.get("coulomb", "simple"),
                cutoff=aimnet_options.get("cutoff", 15.0),
                dsf_alpha=aimnet_options.get("dsf_alpha", 0.2),
                implicit=self.implicit,
                solvent=self.solvent,
                model_options=aimnet_options,
            )
        elif model in AIMNET_PBC_MODELS:
            from .aimnet._aimnet2_official_pbc_calculator import AIMNet2OfficialPBCCalculator

            aimnet_options = self._validated_aimnet_options()
            calculator = AIMNet2OfficialPBCCalculator(
                model=model,
                device=self.device,
                coulomb_method=aimnet_options.get("coulomb", "dsf"),
                cutoff=aimnet_options.get("cutoff", 15.0),
                dsf_alpha=aimnet_options.get("dsf_alpha", 0.2),
                ewald_accuracy=aimnet_options.get("ewald_accuracy", 1e-6),
                pme_cutoff=aimnet_options.get("pme_cutoff"),
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in MACE_PBC_MODELS:
            from .mace._mace_official_pbc_calculator import MACEOfficialPBCCalculator

            mace_options = self._validated_mace_pbc_options()
            calculator = MACEOfficialPBCCalculator(
                model=model,
                device=self.device,
                foundation=mace_options.get("foundation"),
                default_dtype=mace_options.get("default_dtype", "float32"),
                dispersion=mace_options.get("dispersion", False),
                head=mace_options.get("head"),
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in MACEPOL_PBC_MODELS:
            from .mace._macepol_official_pbc_calculator import MACEPolOfficialPBCCalculator

            macepol_options = self._validated_macepol_pbc_options()
            calculator = MACEPolOfficialPBCCalculator(
                model=model,
                device=self.device,
                default_dtype=macepol_options.get("default_dtype", "float32"),
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model == "uma":
            from .uma._uma_calculator import (
                UMACalculator,
                UMA_DEFAULT_SIZE,
                UMA_FALLBACK_HF_MODELS,
            )

            uma_task = self.model_options.get("task")
            uma_size = self.model_options.get("size")
            uma_inference = self.model_options.get("inference")
            effective_device = UMACalculator._normalize_device(self.device)
            uma_inference = self._coerce_uma_inference_for_device(
                uma_inference, effective_device
            )
            checkpoint_path = None
            effective_size = uma_size if uma_size else UMA_DEFAULT_SIZE
            if effective_size in UMA_FALLBACK_HF_MODELS:
                checkpoint = self._ensure_model_file(effective_size)
                checkpoint_path = str(checkpoint) if checkpoint is not None else None

            calculator = UMACalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
                task=uma_task,
                size=uma_size,
                checkpoint_path=checkpoint_path,
                inference_settings=uma_inference,
            )
        elif model == "maceomol":
            self._ensure_model_file(model)
            from .mace._mace_general_calculator import MACEModelCalculator

            calculator = MACEModelCalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"macepols", "macepolm", "macepoll"}:
            from .mace._macepol_calculator import MACEPolCalculator

            model_path = self.model_options.get("model_path")
            if model_path is None:
                downloaded = self._ensure_model_file(model)
                model_path = str(downloaded) if downloaded is not None else None

            calculator = MACEPolCalculator(
                model=model,
                device=self.device,
                model_path=model_path,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        else:
            raise ValueError(f"Model '{model}' is not implemented yet.")

        self._apply_hessian_mode(calculator)
        return calculator

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        if self.model not in IMPLEMENTATION_MODELS:
            self.log_error(f"\n [ERROR] Unsupported model: {self.model}\n")
            raise ValueError(f"Unsupported model: '{self.model}'.")

        self._validated_aimnet_options()
        self._validated_mace_pbc_options()
        self._validated_macepol_pbc_options()
        self._validate_requested_hessian_mode()

        if self.d4 and self.model not in {"ani2x", "ani1x", "ani1ccx", "ani1xnr"}:
            self.log_info([f"\n [WARNING] D4 is not supported for model '{self.model}'. D4 will be ignored.\n"])

        calculator = self._build_calculator()
        self._annotate_calculator_capabilities(calculator)
        _validate_pbc_neighbor_cutoff(self.atoms, calculator)
        self._warn_charge_mult()
        return calculator

    def log_error(self, error_message: str) -> None:
        with open(self.output, "a") as handle:
            handle.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        with open(self.output, "a") as handle:
            for line in info_message:
                handle.write(line)


# Correctly-spelled alias for the (historically misspelled) public class name.
# ``SetClaculator`` stays the canonical definition so existing call sites are not
# churned; ``SetCalculator`` is the preferred spelling for new code.
SetCalculator = SetClaculator


def validate_pbc_capabilities(atoms, task: str) -> None:
    """Reject PBC inputs when the attached calculator lacks real periodic support."""
    if atoms is None or not any(atoms.pbc):
        return

    from maple.function.dispatcher.md.utils import (
        _calc_capability,
        _calc_label,
        _calc_model_name,
    )

    calc = getattr(atoms, "calc", None)
    model_label = _calc_label(calc)

    if (
        _calc_model_name(calc) == "uma"
        and getattr(calc, "_auto_task", True) is False
        and str(getattr(calc, "task_name", "")).lower() == "omol"
    ):
        raise ValueError(
            "PBC is incompatible with UMA task='omol'. "
            "Omit task= so MAPLE can select a periodic UMA task, or set task='omat'."
        )

    if not _calc_capability(calc, "maple_pbc_md_supported"):
        raise ValueError(
            f"{task.upper()} with PBC requires a calculator with real periodic support. "
            f"Model/calculator '{model_label}' is not declared PBC capable."
        )

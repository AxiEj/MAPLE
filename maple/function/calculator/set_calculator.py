from __future__ import annotations

import importlib
import hashlib
import inspect
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import ase
from ase import Atoms

from .calculator_base import (
    ROUTE2_SMD_CALCULATOR_PROFILE,
    CalcABC,
    atoms_has_pbc,
    get_registered_calculator,
    import_calculator_plugin,
    load_calculator_plugins_from_env,
    normalize_none_option,
    validate_implicit_solvent_choice,
)
from .model_capabilities import (
    MODEL_CARD_EXTENSIONS,
    CombinationValidator,
    SolvationCapabilities,
    is_explicit_cpu_device,
    load_model_provenance_card,
)
from .solvation_execution import SolvationExecution

HF_REPO_ID = 'Wayne7815/MAPLE_models'
HF_MODEL_REVISION = 'd5d8eb902246fb3f0e3b24e0d4bbcdc12f12e816'
_COMMON_MODEL_OPTION_KEYS = {'module', 'model_path', 'hessian'}


# Static seed map: builtin name → module that registers the calculator class.
# External users add backends via input-header `module=` or the
# MAPLE_CALCULATOR_PLUGINS env var; this dict is for shipped backends only.
_BUILTIN_NAME_TO_MODULE = {
    "ani2x": "maple.function.calculator.ani._ani_calculator",
    "ani1x": "maple.function.calculator.ani._ani_calculator",
    "ani1ccx": "maple.function.calculator.ani._ani_calculator",
    "ani1xnr": "maple.function.calculator.ani._ani_calculator",
    "aimnet2": "maple.function.calculator.aimnet._aimnet2_calculator",
    "aimnet2nse": "maple.function.calculator.aimnet._aimnet2_calculator",
    "aimnet2-cpcms-v2": "maple.function.calculator.aimnet._aimnet2_cpcms_calculator",
    "mace-off24-medium": "maple.function.calculator.mace._maceoff24_calculator",
    "mace-off23-sc": "maple.function.calculator.mace._maceoff23_sc_calculator",
    "aceff-2.0": "maple.function.calculator.aceff._aceff2_calculator",
    "maceoff23s": "maple.function.calculator.mace._mace_calculator",
    "maceoff23m": "maple.function.calculator.mace._mace_calculator",
    "maceoff23l": "maple.function.calculator.mace._mace_calculator",
    "egret": "maple.function.calculator.mace._mace_calculator",
    "maceomol": "maple.function.calculator.mace._mace_general_calculator",
    "macepols": "maple.function.calculator.mace._macepol_calculator",
    "macepolm": "maple.function.calculator.mace._macepol_calculator",
    "macepoll": "maple.function.calculator.mace._macepol_calculator",
    "fennol": "maple.function.calculator.fennol._fennol_calculator",
    "fennix-bio1s": "maple.function.calculator.fennol._fennol_calculator",
    "fennix-bio1m": "maple.function.calculator.fennol._fennol_calculator",
    "fennix-bio1": "maple.function.calculator.fennol._fennix_bio1_calculator",
    "fennix-bio1-small": "maple.function.calculator.fennol._fennix_bio1_calculator",
    "fennix-bio1-medium": "maple.function.calculator.fennol._fennix_bio1_calculator",
    "uma": "maple.function.calculator.uma._uma_calculator",
    "consolv": "maple.function.calculator.consolv._consolv_calculator",
}


_plugins_loaded_from_env = False


def _compact_model_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace('_', '')
        .replace('-', '')
        .replace('.', '')
        .replace(' ', '')
        .replace('(', '')
        .replace(')', '')
    )


_BUILTIN_ALIAS_TO_NAME = {
    _compact_model_name(name): name for name in _BUILTIN_NAME_TO_MODULE
}


def _builtin_canonical_name(name: str) -> Optional[str]:
    return _BUILTIN_ALIAS_TO_NAME.get(_compact_model_name(name))


def _model_download_url(filename: str) -> str:
    import os

    revision = os.environ.get('MAPLE_MODEL_REVISION', HF_MODEL_REVISION).strip() or HF_MODEL_REVISION
    return f'https://huggingface.co/{HF_REPO_ID}/resolve/{revision}/{filename}'


def _normalize_model_options(model_options: Optional[dict]) -> dict:
    """Normalize option keys and enum-like values without touching path values."""
    options = {}
    for raw_key, raw_value in (model_options or {}).items():
        key = str(raw_key).strip().lower()
        if raw_value is None:
            value = None
        elif key in {'hessian', 'coulomb_method', 'inference', 'task', 'size'}:
            value = str(raw_value).strip().lower()
        elif key == "model_path" and not isinstance(raw_value, str):
            value = raw_value
        elif key in {'module', 'model_path', 'checkpoint_path'}:
            value = str(raw_value).strip()
        elif isinstance(raw_value, str):
            value = raw_value.strip()
        else:
            value = raw_value
        options[key] = value
    return options


class SetCalculator:

    def __init__(
        self,
        device,
        model: str,
        output: str,
        atoms: Optional[Atoms] = None,
        d4: bool = False,
        implicit: str = "None",
        solvent: str = "None",
        model_options: Optional[dict] = None,
        solvation_options: Optional[dict] = None,
        charge_options: Optional[dict] = None,
        task: Optional[str] = None,
    ) -> None:
        self.output = output
        self.model = str(model).strip().lower()
        self.d4 = d4
        self.device = device
        self.atoms = atoms
        self.implicit = normalize_none_option(implicit)
        self.solvent = normalize_none_option(solvent)
        self.model_options = _normalize_model_options(model_options)
        info = atoms.info if atoms is not None else {}
        self.solvation_options = dict(info.get('_maple_solvation_options', {}))
        self.solvation_options.update(solvation_options or {})
        self.charge_options = dict(info.get('_maple_charge_options', {}))
        self.charge_options.update(charge_options or {})
        self.task = str(task).strip().lower() if task is not None else None
        self._model_error_logged = False

    def _model_dir(self) -> Path:
        return Path(__file__).parent / 'model'

    def _model_dir_description(self) -> str:
        model_dir = self._model_dir()
        resolved_dir = model_dir.resolve()
        if resolved_dir != model_dir:
            return f'{model_dir} (resolved: {resolved_dir})'
        return str(model_dir)

    def _log_model_error(self, message: str) -> None:
        self._model_error_logged = True
        self.log_error(message)

    def _model_card_root(self) -> Path:
        return Path(__file__).parent / "model_cards"

    def _registered_model_card(self, cls):
        names = tuple(getattr(cls, "MODEL_NAMES", ()))
        if not names:
            return None
        name = str(names[0]).strip().lower().replace("_", "-")
        root = self._model_card_root()
        if not any((root / f"{name}{ext}").is_file() for ext in MODEL_CARD_EXTENSIONS):
            return None
        return load_model_provenance_card(name, root)

    def _validate_registered_model_task(self, cls) -> None:
        card = self._registered_model_card(cls)
        if card is not None and self.task is not None:
            card.validate_task(self.task)

    def _validate_registered_model_acceleration(self, cls) -> None:
        """Reject unverified accelerator paths before resolving any weights."""
        if self.implicit == "anysolv" and not is_explicit_cpu_device(self.device):
            self._validate_anysolv_composite_acceleration(cls)
            return
        if not getattr(cls, "MODEL_NAMES", ()) and not is_explicit_cpu_device(self.device):
            raise ValueError("Accelerator execution requires a stable MODEL_NAMES identity.")
        card = self._registered_model_card(cls)
        if card is None:
            if is_explicit_cpu_device(self.device):
                return
            raise ValueError(
                f"Accelerator execution for '{self.model}' is disabled: no frozen "
                "no-loss CPU/GPU parity evidence card is registered."
            )
        if self.model == "consolv":
            SolvationExecution(
                self.device,
                self.model_options.get("dtype", "float32"),
                self.model_options.get("inference", "reference"),
            ).validate_card(card, task=self.task or "sp")
            return
        card.validate_device(
            self.device,
            task=self.task,
            inference_mode=self.model_options.get("inference"),
            evidence_root=Path(__file__).resolve().parents[3],
        )

    def _validate_anysolv_composite_acceleration(self, cls) -> None:
        """Qualify the scalar composition, not standalone AIMNet2 mechanics."""
        if (
            self.model != "aimnet2"
            or cls.__module__ != _BUILTIN_NAME_TO_MODULE["aimnet2"]
            or not issubclass(cls, CalcABC)
        ):
            raise ValueError(
                "CUDA AnySolv composition is qualified only for the exact AIMNet2 scalar base."
            )
        if (
            self.d4
            or str(self.model_options.get("coulomb_method", "simple")).lower()
            != "simple"
        ):
            raise ValueError(
                "The qualified CUDA AnySolv composition requires the unmodified AIMNet2 simple-Coulomb base, without D4."
            )
        card = load_model_provenance_card("aimnet2-anysolv", self._model_card_root())
        card.validate_task(self.task or "sp")
        execution = SolvationExecution(
            self.device,
            self.solvation_options.get("dtype", "float32"),
            self.solvation_options.get("inference", "reference"),
        )
        execution.validate_card(card, task=self.task or "sp")
        self._anysolv_composite_card = card

    def _validate_anysolv_base_checkpoint(self, checkpoint) -> None:
        card = getattr(self, "_anysolv_composite_card", None)
        if card is None:
            return
        expected = card.payload["checkpoints"]["gas"]["checkpoint_sha256"]
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "The qualified CUDA AnySolv composition requires its pinned local AIMNet2 checkpoint."
            )
        actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                "AIMNet2 checkpoint does not match the qualified CUDA AnySolv composition."
            )

    def _validate_solvent_config(self) -> None:
        self.implicit, self.solvent = validate_implicit_solvent_choice(
            self.implicit, self.solvent
        )
        if self.implicit == 'none':
            return

        if self.implicit in {"gb", "pb", "smd"}:
            for key, expected in (
                ("method", self.implicit),
                ("implicit", self.solvent),
                ("solvent", self.solvent),
            ):
                if key in self.solvation_options and normalize_none_option(
                    self.solvation_options[key]
                ) != expected:
                    raise ValueError(f"Conflicting implicit-solvation {key} selectors.")
            self.solvation_options.setdefault("method", self.implicit)
            self.solvation_options.setdefault("implicit", self.solvent)
            if self.solvent != "water":
                raise ValueError("Modern implicit solvation supports solvent='water' only.")
            if self.solvation_options.get("experimental") is not True:
                raise ValueError("Implicit-solvation providers require experimental=true.")
            mode = self.model_options.get("hessian")
            if mode not in (None, "numerical") or (
                self.task in {"freq", "frequency"} and mode != "numerical"
            ):
                raise ValueError("Implicit solvation supports a numerical Hessian only.")
            if self.atoms is None:
                raise ValueError("Implicit solvation requires an input structure.")
            if atoms_has_pbc(self.atoms):
                raise ValueError("Implicit solvation is non-periodic; remove #pbc.")
            if self.implicit == "smd" and (
                self.task not in (None, "sp") or mode is not None or self.charge_options
            ):
                raise ValueError("Route 2 SMD is single-point energy-only and does not accept #charge.")
            return

        if self.implicit == "anysolv":
            allowed = {
                "method",
                "implicit",
                "experimental",
                "model_path",
                "source_root",
                "dtype",
                "inference",
            }
            extra = set(self.solvation_options) - allowed
            if extra:
                raise ValueError(
                    f"AnySolv does not accept options: {', '.join(sorted(extra))}."
                )
            if self.solvation_options.get("method", "anysolv") != "anysolv":
                raise ValueError("Conflicting AnySolv method selectors.")
            if self.solvation_options.get("implicit", self.solvent) != self.solvent:
                raise ValueError("Conflicting AnySolv solvent selectors.")
            if self.solvation_options.get("experimental") is not True:
                raise ValueError("AnySolv-MoE requires experimental=true.")
            if (
                self.task not in (None, "sp")
                or self.model_options.get("hessian") is not None
            ):
                raise ValueError(
                    "AnySolv-MoE supports scalar single-point energy only."
                )
            if self.d4 or self.charge_options:
                raise ValueError(
                    "AnySolv does not compose D4 or a partial-charge provider."
                )
            source_root = self.solvation_options.get("source_root")
            if not isinstance(source_root, str) or not source_root.strip():
                raise ValueError(
                    "AnySolv requires source_root=<non-empty pinned official checkout path>."
                )
            if "model_path" in self.solvation_options:
                model_path = self.solvation_options["model_path"]
                if not isinstance(model_path, str) or not model_path.strip():
                    raise ValueError(
                        "AnySolv model_path must be a non-empty path string."
                    )
            if self.atoms is not None and atoms_has_pbc(self.atoms):
                raise ValueError("AnySolv-MoE is non-periodic; remove #pbc.")
            return

        if self.implicit != 'gbsa':
            raise ValueError(
                "Unsupported implicit solvation method: "
                f"'{self.implicit}'. Supported experimental method: gbsa."
            )

        if self.solvation_options.get('experimental') is not True:
            raise ValueError(
                "Implicit GB-polar/QEq solvation is experimental and disabled "
                "by default. Add experimental=true in #solv(...) to request "
                "energy-only use."
            )

        if self.model_options.get('hessian') is not None:
            raise ValueError(
                "Experimental implicit GB-polar solvation does not support "
                "Hessian/HVP workflows."
            )

        if self.atoms is not None and atoms_has_pbc(self.atoms):
            raise ValueError(
                "Experimental implicit GB-polar solvation is non-periodic only; "
                "remove #pbc or use a periodic solvent backend."
            )

    def _discover_calculator_class(self, name: str):
        """Resolve `name` to a registered calculator class.

        Plug-in discovery layers:
        1. Explicit `module=` in model_options (run import_calculator_plugin).
        2. _BUILTIN_NAME_TO_MODULE seed for shipped backends.
        3. MAPLE_CALCULATOR_PLUGINS env var (loaded once per process).
        """
        global _plugins_loaded_from_env
        if not _plugins_loaded_from_env:
            load_calculator_plugins_from_env()
            _plugins_loaded_from_env = True

        normalized = str(name).strip().lower()
        module_override = self.model_options.get('module')
        if module_override:
            import_calculator_plugin(str(module_override))
            resolved_name = normalized
        else:
            builtin_name = _builtin_canonical_name(normalized)
            builtin_module = _BUILTIN_NAME_TO_MODULE.get(normalized)
            if builtin_module is None and builtin_name is not None:
                builtin_module = _BUILTIN_NAME_TO_MODULE.get(builtin_name)
            if builtin_module is not None:
                importlib.import_module(builtin_module)
            # Implicit builtin discovery must not let a cached compact plugin
            # name hijack a declared builtin alias. Explicit modules above keep
            # the requested registry identity instead.
            resolved_name = builtin_name or normalized

        try:
            cls = get_registered_calculator(resolved_name)
            self.model = resolved_name
            return cls
        except KeyError:
            pass

        raise ValueError(f"Unsupported model: '{name}'.")

    def _prepare_solvent_correction(self, cls):
        """Validate composition and actual provider capabilities before weights."""
        if not issubclass(cls, CalcABC):
            raise TypeError("Modern implicit solvation requires the CalcABC finalizer.")
        if self.implicit == "smd" and (
            not callable(getattr(cls, "polar_state", None))
            or getattr(cls, "route2_smd_profile", None) != ROUTE2_SMD_CALCULATOR_PROFILE
        ):
            raise TypeError(
                "SMD requires the declared Route 2 polar_state() density/response API "
                "and compatible calculator profile; this backend does not supply it."
            )

        mode = self.model_options.get("hessian")
        card = self._registered_model_card(cls)
        if mode == "numerical":
            # This is a required scientific permission, not optional discovery.
            names = tuple(getattr(cls, "MODEL_NAMES", ()))
            if not names:
                raise ValueError("Numerical solvent Hessian requires a stable MODEL_NAMES identity.")
            card = load_model_provenance_card(names[0], self._model_card_root())
            expected_identity = str(names[0]).strip().lower().replace("_", "-")
            if card.model_id != expected_identity:
                raise ValueError("Numerical solvent Hessian model-card identity mismatch.")
            card.validate_task("frequency")
        if card is not None:
            CombinationValidator(
                model_capabilities=card.capabilities,
                implicit=self.implicit,
                task=self.task or "sp",
            ).validate()

        from .extra_correction.implicit import ImplicitSolvationCorrection

        correction = ImplicitSolvationCorrection(
            self.atoms,
            charge_options=self.charge_options,
            solvation_options=self.solvation_options,
            output=self.output,
        )
        capabilities = SolvationCapabilities.from_backend(correction)
        if self.task in {"opt", "scan", "ts", "irc", "md", "freq", "frequency"} and not (
            capabilities.energy and capabilities.forces and capabilities.conservative_forces
        ):
            raise ValueError("The requested task requires an energy-consistent solvent force backend.")
        profile = None
        if card is not None:
            profile = CombinationValidator(
                model_capabilities=card.capabilities,
                implicit=self.implicit,
                hessian_mode=mode,
                solvation_capabilities=capabilities,
            ).validate()
        return correction, profile

    def _validate_against_class(self, cls) -> None:
        """Pre-instantiation gates: pbc, hessian mode, charge/mult, d4."""
        if self.atoms is not None and atoms_has_pbc(self.atoms) and not getattr(cls, 'SUPPORTS_PBC', False):
            raise NotImplementedError(
                f"Model '{self.model}' is a no-PBC molecular wrapper. "
                "Use UMA or a backend-native PBC calculator for periodic systems."
            )

        mode = self.model_options.get('hessian')
        if mode is not None:
            if mode not in cls.SUPPORTED_HESSIAN_MODES:
                supported_text = ', '.join(sorted(cls.SUPPORTED_HESSIAN_MODES))
                raise ValueError(
                    f"Model '{self.model}' does not support hessian='{mode}'. "
                    f'Supported modes: {supported_text}'
                )

        coulomb_method = self.model_options.get('coulomb_method')
        supported_coulomb = getattr(cls, 'SUPPORTED_COULOMB_METHODS', None)
        if coulomb_method is not None and supported_coulomb is not None:
            if coulomb_method not in supported_coulomb:
                supported_text = ', '.join(sorted(supported_coulomb))
                if coulomb_method == 'ewald':
                    raise NotImplementedError(
                        "AIMNet2 coulomb_method='ewald' requires validated PBC/cell/MIC support; "
                        f"use one of: {supported_text}."
                    )
                raise ValueError(
                    f"Model '{self.model}' does not support coulomb_method='{coulomb_method}'. "
                    f'Supported methods: {supported_text}'
                )

        if self.atoms is not None:
            has_charge = self.atoms.info.get('charge', 0) != 0
            has_mult = self.atoms.info.get('mult', 1) != 1
            if (has_charge or has_mult) and not cls.SUPPORTS_CHARGE_MULT:
                self.log_info(
                    [
                        f"\n [WARNING] Model '{self.model}' does not support charge/multiplicity.\n",
                        f"           charge={self.atoms.info.get('charge', 0)}, ",
                        f"mult={self.atoms.info.get('mult', 1)} will be IGNORED.\n",
                        '           Models with charge/mult support: aimnet2, aimnet2nse, uma, macepols/m/l\n',
                    ]
                )

        if self.d4:
            import inspect

            ctor_params = inspect.signature(cls.__init__).parameters
            if 'd4' not in ctor_params:
                self.log_info(
                    [f"\n [WARNING] D4 is not supported for model '{self.model}'. D4 will be ignored.\n"]
                )

    def _validate_model_options(self, cls) -> None:
        """Fail loudly on misspelled/unsupported model_options for shipped backends."""
        if self.model == "consolv" and "model_path" in self.model_options:
            path = self.model_options["model_path"]
            if not isinstance(path, str) or not path.strip():
                raise ValueError("ConSolv model_path must be a non-empty path string.")
        option_keys = getattr(cls, 'OPTION_KEYS', None)
        if option_keys is None:
            return
        allowed = set(_COMMON_MODEL_OPTION_KEYS)
        allowed.update(option_keys)
        unknown = sorted(key for key in self.model_options if key not in allowed)
        if unknown:
            allowed_text = ', '.join(sorted(allowed)) or '(none)'
            raise ValueError(
                f"Unsupported model option(s) for '{self.model}': {', '.join(unknown)}. "
                f"Supported options: {allowed_text}"
            )

    def _explicit_model_path_option(self, cls) -> Optional[str]:
        """Return the ctor kwarg that consumes model_path, or None if unsupported."""
        option = getattr(cls, 'MODEL_PATH_OPTION', None)
        if option:
            return str(option)

        # Compatibility for external plugins that have not adopted the class
        # attribute yet but expose the conventional constructor kwarg.
        try:
            ctor_params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            return None
        if 'model_path' in ctor_params:
            return 'model_path'
        if 'checkpoint_path' in ctor_params:
            return 'checkpoint_path'
        return None

    def _resolve_explicit_model_path(self, cls, options: dict) -> Path:
        """Validate an explicit user model_path and prevent silent ignore."""
        path_option = self._explicit_model_path_option(cls)
        if path_option is None:
            raise ValueError(
                f"Model '{self.model}' does not support model_path; remove the option "
                "or use a backend with an explicit model_path/checkpoint_path input."
            )

        if path_option == 'checkpoint_path' and options.get('checkpoint_path'):
            explicit_model_path = Path(str(options['model_path'])).expanduser().resolve()
            explicit_checkpoint_path = Path(str(options['checkpoint_path'])).expanduser().resolve()
            if explicit_model_path != explicit_checkpoint_path:
                raise ValueError(
                    "Specify only one of model_path or checkpoint_path for this backend; "
                    "the two paths differ."
                )

        resolved_model_path = Path(str(options['model_path'])).expanduser()
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Explicit model_path for '{self.model}' does not exist or is not a file: "
                f"{resolved_model_path}"
            )
        return resolved_model_path

    def _resolve_model_path(self, cls, name: str) -> Optional[Path]:
        """Resolve checkpoint path per class attrs.

        - If `cls.CHECKPOINT_FILENAME` maps the name → ensure (download or local).
        - Else if `cls.REQUIRES_LOCAL_MODEL_FILE` is True → require local.
        - Else → no path (backend looks up its own default).
        """
        local_filenames = getattr(cls, 'LOCAL_MODEL_FILENAMES', None)
        if local_filenames and name in local_filenames:
            return self._require_local_model_file(name, local_filenames[name])
        if cls.CHECKPOINT_FILENAME and name in cls.CHECKPOINT_FILENAME:
            filename = cls.CHECKPOINT_FILENAME[name]
            return self._ensure_model_file(filename, name)
        if cls.REQUIRES_LOCAL_MODEL_FILE:
            return self._require_local_model_file(name)
        return None

    def _coerce_uma_inference_for_device(self, inference, device_name: str):
        if inference == 'turbo' and device_name == 'cpu':
            self.log_info(
                [
                    "\n [WARNING] UMA inference='turbo' requires CUDA; "
                    "falling back to 'default' on CPU.\n"
                ]
            )
            return 'default'
        return inference

    def _ensure_model_file(self, filename: str, model_name: str) -> Path:
        """Ensure a HF-hosted checkpoint is on disk; download if missing.

        Production safeguards: a pinned HuggingFace revision (override with
        MAPLE_MODEL_REVISION only when intentionally refreshing model assets),
        an offline switch (MAPLE_OFFLINE), a socket timeout
        (MAPLE_DOWNLOAD_TIMEOUT, default 60s), a process-unique temp file so
        concurrent runs never clobber each other's partial download, and a
        byte-count check against Content-Length before the atomic replace so a
        truncated stream cannot land as the final model file.
        Checksum pinning is intentionally omitted: it needs an upstream
        hash manifest, which this repo does not publish — do not fake one.
        """
        import os

        model_dir = self._model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if model_path.exists():
            return model_path

        if os.environ.get('MAPLE_OFFLINE', '').strip().lower() in ('1', 'true', 'yes', 'on'):
            message = (
                f"Model file '{filename}' for '{model_name}' is missing and MAPLE_OFFLINE "
                'is set, so auto-download is disabled.\n'
                f'       MAPLE model directory searched: {self._model_dir_description()}'
            )
            self._log_model_error(message)
            raise FileNotFoundError(message)

        url = _model_download_url(filename)
        self.log_info(
            [
                f" [INFO] Model file '{filename}' not found locally.\n",
                f' [INFO] Downloading from: {url}\n',
            ]
        )

        timeout = float(os.environ.get('MAPLE_DOWNLOAD_TIMEOUT', '60'))
        temp_path = model_path.with_suffix(f'.{os.getpid()}.tmp')
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                size = response.headers.get('Content-Length')
                expected = int(size) if size else None
                if expected is not None:
                    self.log_info([f' [INFO] File size: {expected / 1024 / 1024:.1f} MB\n'])
                with open(temp_path, 'wb') as handle:
                    shutil.copyfileobj(response, handle)
            downloaded = temp_path.stat().st_size
            if expected is not None and downloaded != expected:
                temp_path.unlink(missing_ok=True)
                message = (
                    f"Incomplete download for model '{model_name}': received {downloaded} bytes, "
                    f'expected {expected} (Content-Length).'
                )
                self._log_model_error(message)
                raise RuntimeError(message)
            temp_path.replace(model_path)
            self.log_info([f' [INFO] Download complete: {model_path}\n'])
        except urllib.error.HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            self._log_model_error(
                f"Download failed for model '{model_name}' (HTTP {exc.code}): {url}\n"
                f'       MAPLE model directory: {self._model_dir_description()}'
            )
            raise RuntimeError(f"Failed to download model '{model_name}': HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            temp_path.unlink(missing_ok=True)
            reason = getattr(exc, 'reason', exc)
            self._log_model_error(
                f"Network error while downloading model '{model_name}': {reason}\n"
                f'       MAPLE model directory: {self._model_dir_description()}'
            )
            raise RuntimeError(f"Failed to download model '{model_name}': {reason}") from exc

        return model_path

    def _local_model_file(self, filename: str) -> Optional[Path]:
        model_path = self._model_dir() / filename
        return model_path if model_path.exists() else None

    def _require_local_model_file(self, model_name: str, filename: Optional[str] = None) -> Path:
        filename = filename or f'{model_name}.pt'
        model_path = self._model_dir() / filename
        if model_path.exists():
            return model_path

        message = (
            f"Model file '{filename}' for '{model_name}' was not found locally.\n"
            f'       MAPLE model directory searched: {self._model_dir_description()}\n'
            f'       Expected file path: {model_path}\n'
            f'This model is not available from {HF_REPO_ID}; install the backend-specific model '
            'file or pass an explicit model_path when supported.'
        )
        self._log_model_error(message)
        raise FileNotFoundError(message)

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        requested_name = self.model
        self._validate_solvent_config()

        cls = self._discover_calculator_class(requested_name)
        name = self.model
        self._validate_registered_model_task(cls)
        self._validate_model_options(cls)
        self._validate_registered_model_acceleration(cls)
        self._validate_against_class(cls)
        correction = profile = None
        if self.implicit in {"gb", "pb", "smd"}:
            correction, profile = self._prepare_solvent_correction(cls)
        if self.implicit == "anysolv":
            # Keep composition at CalcABC's single result boundary.  UMA uses
            # a separate path and is deliberately not wrapped or special-cased.
            expected_module = _BUILTIN_NAME_TO_MODULE.get(name)
            if not issubclass(cls, CalcABC) or cls.__module__ != expected_module:
                raise ValueError("AnySolv requires a built-in vacuum CalcABC model.")
            card = self._registered_model_card(cls)
            if card is not None and card.capabilities.solvation_mode != "none":
                raise ValueError(
                    "AnySolv cannot be added to a native solution-phase model."
                )
            if self.atoms is not None and not cls.SUPPORTS_CHARGE_MULT:
                if (
                    self.atoms.info.get("charge", 0) != 0
                    or self.atoms.info.get("mult", 1) != 1
                ):
                    raise ValueError(
                        "The selected vacuum base cannot represent this charge/multiplicity."
                    )
        options = dict(self.model_options)
        options.setdefault('d4', self.d4)
        # Allow input header to override the auto-resolved model path.
        if options.get('model_path'):
            resolved_model_path = self._resolve_explicit_model_path(cls, options)
        else:
            resolved_model_path = self._resolve_model_path(cls, name)
        self._validate_anysolv_base_checkpoint(resolved_model_path)

        # UMA's `size`/`checkpoint_path` resolution is special: when no checkpoint
        # is given and the requested size has a known HF fallback, prefer a
        # locally-cached MAPLE copy if present. Handle the coercion here so that
        # UMACalculator only receives clean kwargs.
        if cls.__name__ == 'UMACalculator':
            from .uma._uma_calculator import (
                UMA_DEFAULT_SIZE,
                UMA_FALLBACK_HF_MODELS,
                UMACalculator,
            )

            inference = options.get('inference')
            effective_device = UMACalculator._normalize_device(self.device)
            options['inference'] = self._coerce_uma_inference_for_device(inference, effective_device)

            checkpoint_path = options.get('checkpoint_path')
            if checkpoint_path:
                checkpoint_path = Path(str(checkpoint_path)).expanduser()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Explicit checkpoint_path for '{self.model}' does not exist or is not a file: "
                        f"{checkpoint_path}"
                    )
                checkpoint_path = str(checkpoint_path)
            elif resolved_model_path is not None:
                checkpoint_path = str(resolved_model_path)
            effective_size = str(options.get('size')).lower() if options.get('size') else UMA_DEFAULT_SIZE
            if checkpoint_path is None and effective_size in UMA_FALLBACK_HF_MODELS:
                local_checkpoint = self._local_model_file(f'{effective_size}.pt')
                if local_checkpoint is not None:
                    checkpoint_path = str(local_checkpoint)
            options['checkpoint_path'] = checkpoint_path

        resolved_path_str = str(resolved_model_path) if resolved_model_path is not None else None
        kwargs = cls.build_kwargs_from_options(name, options, resolved_model_path=resolved_path_str)
        if "execution_task" in inspect.signature(cls.__init__).parameters:
            kwargs["execution_task"] = self.task

        calculator: Any = cls(
            device=self.device,
            model=name,
            implicit="none" if self.implicit == "anysolv" else self.implicit,
            solvent="none" if self.implicit == "anysolv" else self.solvent,
            **kwargs,
        )

        if correction is not None:
            if getattr(calculator, "solvent_correction", None) is not None:
                raise ValueError("A second solvent correction cannot be attached.")
            calculator.solvent_correction = correction
            if profile is not None:
                calculator.frequency_type = profile.frequency_type

        if self.implicit == "anysolv":
            from .extra_correction.implicit import AnySolvBackend

            if getattr(calculator, "solvent_correction", None) is not None:
                raise ValueError("A second solvent correction cannot be attached.")
            configure = getattr(calculator, "_set_scalar_composite_execution", None)
            if name == "aimnet2" and callable(configure):
                configure(
                    SolvationExecution(
                        self.device,
                        self.solvation_options.get("dtype", "float32"),
                        "reference",
                    )
                )
            calculator.solvent_correction = AnySolvBackend(
                solvent=self.solvent,
                source_root=self.solvation_options["source_root"],
                device=str(self.device),
                dtype=self.solvation_options.get("dtype", "float32"),
                inference=self.solvation_options.get("inference", "reference"),
                execution_task=self.task,
                **(
                    {"model_path": self.solvation_options["model_path"]}
                    if "model_path" in self.solvation_options
                    else {}
                ),
            )
            calculator.implemented_properties = [
                prop
                for prop in calculator.implemented_properties
                if prop in {"energy", "free_energy"}
            ]

        self._apply_hessian_mode(calculator)
        return calculator

    def _apply_hessian_mode(self, calculator) -> None:
        mode = self.model_options.get('hessian')
        if mode is None:
            return

        supported = type(calculator).SUPPORTED_HESSIAN_MODES
        if mode not in supported:
            supported_text = ', '.join(sorted(supported))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f'Supported modes: {supported_text}'
            )

        if not hasattr(calculator, 'hessian'):
            raise ValueError(f"Model '{self.model}' does not expose configurable Hessian modes.")

        calculator.hessian = mode

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        try:
            calculator = self._build_calculator()
            return calculator
        except Exception as exc:
            if not self._model_error_logged:
                self._log_model_error(
                    f"Failed to initialize model '{self.model}'.\n"
                    f'       MAPLE model directory: {self._model_dir_description()}\n'
                    f'       Error: {type(exc).__name__}: {exc}'
                )
            raise

    def log_error(self, error_message: str) -> None:
        with open(self.output, 'a') as handle:
            handle.write(f'ERROR: {error_message.rstrip()}\n')

    def log_info(self, info_message: list) -> None:
        with open(self.output, 'a') as handle:
            for line in info_message:
                handle.write(line)


# Historical misspelling kept as an alias for one release.
SetClaculator = SetCalculator

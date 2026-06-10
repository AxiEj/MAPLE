from __future__ import annotations

import importlib
import inspect
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import ase
import numpy as np
from ase import Atoms

from .calculator_base import (
    atoms_has_pbc,
    get_registered_calculator,
    import_calculator_plugin,
    load_calculator_plugins_from_env,
    normalize_none_option,
    validate_implicit_solvent_choice,
)


HF_REPO_ID = 'Wayne7815/MAPLE_models'
HF_MODEL_REVISION = 'd5d8eb902246fb3f0e3b24e0d4bbcdc12f12e816'
_COMMON_MODEL_OPTION_KEYS = {'module', 'model_path', 'hessian'}


# Static seed map: builtin name → module that registers the calculator class.
# External users add backends via input-header `module=` or the
# MAPLE_CALCULATOR_PLUGINS env var; this dict is for shipped backends only.
_BUILTIN_NAME_TO_MODULE = {
    'ani2x': 'maple.function.calculator.ani._ani_calculator',
    'ani1x': 'maple.function.calculator.ani._ani_calculator',
    'ani1ccx': 'maple.function.calculator.ani._ani_calculator',
    'ani1xnr': 'maple.function.calculator.ani._ani_calculator',
    'aimnet2': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2nse': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2-pbc': 'maple.function.calculator.aimnet._aimnet2_official_pbc_calculator',
    'aimnet2nse-pbc': 'maple.function.calculator.aimnet._aimnet2_official_pbc_calculator',
    'maceoff23s': 'maple.function.calculator.mace._mace_calculator',
    'maceoff23m': 'maple.function.calculator.mace._mace_calculator',
    'maceoff23l': 'maple.function.calculator.mace._mace_calculator',
    'egret': 'maple.function.calculator.mace._mace_calculator',
    'mace-mp-pbc-small': 'maple.function.calculator.mace._mace_official_pbc_calculator',
    'mace-mp-pbc-medium': 'maple.function.calculator.mace._mace_official_pbc_calculator',
    'mace-mp-pbc-large': 'maple.function.calculator.mace._mace_official_pbc_calculator',
    'maceomol': 'maple.function.calculator.mace._mace_general_calculator',
    'macepol-pbc-small': 'maple.function.calculator.mace._macepol_official_pbc_calculator',
    'macepol-pbc-medium': 'maple.function.calculator.mace._macepol_official_pbc_calculator',
    'macepol-pbc-large': 'maple.function.calculator.mace._macepol_official_pbc_calculator',
    'macepols': 'maple.function.calculator.mace._macepol_calculator',
    'macepolm': 'maple.function.calculator.mace._macepol_calculator',
    'macepoll': 'maple.function.calculator.mace._macepol_calculator',
    'uma': 'maple.function.calculator.uma._uma_calculator',
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

_plugins_loaded_from_env = False


def _compact_model_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace('_', '')
        .replace('-', '')
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


def _calculator_requires_single_image_mic(calculator) -> bool:
    """Return whether MAPLE must enforce cutoff < Wigner-Seitz inradius.

    The strict MIC gate is mandatory for calculators whose periodic semantics
    are a single-image/local pair graph.  Official periodic backends can instead
    declare ``maple_periodic_neighborlist_multi_image_safe=True`` (or explicitly
    ``maple_requires_single_image_mic=False``) to state that their own periodic
    neighbor list handles replicated images in primitive cells.
    """
    explicit = getattr(calculator, "maple_requires_single_image_mic", None)
    if explicit is not None:
        return bool(explicit)
    return not bool(getattr(calculator, "maple_periodic_neighborlist_multi_image_safe", False))


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


def validate_pbc_cell_geometry(atoms) -> None:
    """Validate the periodic cell geometry on the shared MD admission path.

    MAPLE's wrap/unwrap (``get_unwrapped_positions`` and the
    ``dot(scaled + image_flags, cell.array)`` reconstruction) goes through the
    *full* 3x3 cell matrix.  A rank-deficient full cell — e.g. ``pbc=[T,T,F]``
    paired with a zero z lattice vector — would silently drop the non-periodic
    coordinate even when the active 2-D sublattice is itself valid.  So any
    periodic MD system must carry a finite, rank-3 full cell matrix; full 3-D
    PBC must additionally enclose a positive volume.  The active-sublattice
    finiteness/independence and the minimum-image sanity check are delegated to
    :func:`_minimum_image_radius_A` (which Minkowski-reduces the active lattice),
    so this validator does not duplicate that logic.

    Note this requires ``rank == 3`` for *all* periodic systems, not
    ``rank == sum(pbc)``: a real slab keeps a finite (vacuum) third lattice
    vector and so is rank 3 and accepted, while a genuinely 2-D cell with a zero
    third vector is rank 2 and rejected.
    """
    if atoms is None or not any(atoms.pbc):
        return

    cell = np.asarray(atoms.get_cell(), dtype=float)
    if not np.all(np.isfinite(cell)):
        raise ValueError("PBC MD requires finite cell vectors.")
    # Linear matrix rank, not ASE's length-based ``Cell.rank``: a third lattice
    # vector that is nonzero but linearly dependent (e.g. an in-plane duplicate)
    # is length-rank 3 yet matrix-rank 2, and would still make the full-cell
    # inverse singular inside the wrap/unwrap reconstruction.  Reject it here with
    # a clear message instead of a downstream LinAlgError.
    if np.linalg.matrix_rank(cell) != 3:
        raise ValueError(
            "PBC MD requires a full rank-3 cell matrix. MAPLE reconstructs "
            "unwrapped coordinates through the full 3x3 cell, so a rank-deficient "
            "cell (a periodic axis paired with a zero or linearly dependent "
            "lattice vector) would silently drop a coordinate. Provide three "
            "linearly independent lattice vectors (a slab's vacuum direction "
            "still needs a finite, independent vector)."
        )

    if all(atoms.pbc):
        volume = float(atoms.get_volume())
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError(
                f"Full 3-D PBC MD requires a finite, positive cell volume, got {volume!r}."
            )

    # Active-sublattice finiteness/independence + minimum-image sanity.
    _minimum_image_radius_A(atoms)


def validate_pbc_neighbor_cutoff(
    atoms,
    calculator,
    *,
    allow_unknown_cutoff: bool = False,
    require_known_cutoff: bool = True,
) -> None:
    """Validate cutoff against the true minimum-image radius.

    The cutoff bound follows Allen & Tildesley, Computer Simulation of Liquids,
    2nd ed. (Oxford University Press, 2017), section 1.5: each pair interaction
    should see at most one periodic image.  For skew cells this requires the
    Wigner-Seitz inradius, i.e. half the shortest non-zero periodic lattice
    vector, rather than half the shortest stored cell edge.

    ``require_known_cutoff`` (the MD admission path) rejects a PBC calculator that
    does not expose a cutoff when MAPLE must enforce the single-image MIC
    contract, because the minimum-image convention then cannot be verified;
    ``allow_unknown_cutoff`` is the explicit per-gate escape hatch.  Calculators
    that declare a multi-image-safe periodic neighbor list are outside this MIC
    supercell-only scope and are not rejected merely because their cutoff reaches
    beyond the Wigner-Seitz inradius.  The general ``SetCalculator`` build path
    passes ``require_known_cutoff=False`` so non-MD periodic tasks (single point /
    OPT / SCAN / TS) keep their prior behaviour and are not blocked on an
    undeclared cutoff.
    """
    if atoms is None or not any(atoms.pbc):
        return
    if not getattr(calculator, "maple_pbc_md_supported", False):
        return

    requires_single_image_mic = _calculator_requires_single_image_mic(calculator)
    cutoff = _calculator_neighbor_cutoff_A(calculator)
    if cutoff is None or not np.isfinite(cutoff) or cutoff <= 0.0:
        if require_known_cutoff and requires_single_image_mic and not allow_unknown_cutoff:
            raise ValueError(
                "PBC MD calculator does not expose a neighbor cutoff "
                "(neighbor_cutoff_A / cutoff_A / maple_neighbor_cutoff), so the "
                "minimum-image convention cannot be verified: a cutoff larger than "
                "half the shortest periodic lattice vector would let pair "
                "interactions see multiple periodic images. Use a calculator that "
                "declares its cutoff, or set allow_unknown_cutoff=true to run anyway "
                "(recorded in the run manifest)."
            )
        return

    if not requires_single_image_mic:
        return

    radius, shortest_lattice_vector = _minimum_image_radius_A(atoms)
    if cutoff >= radius:
        raise ValueError(
            f"Backend neighbor cutoff {cutoff:.3f} A >= minimum-image radius "
            f"{radius:.3f} A (shortest periodic lattice vector = "
            f"{shortest_lattice_vector:.3f} A). "
            "Minimum-image convention is violated; enlarge the cell or reduce the cutoff."
        )



def _normalize_model_options(model_options: Optional[dict]) -> dict:
    """Normalize option keys and enum-like values without touching path values."""
    options = {}
    for raw_key, raw_value in (model_options or {}).items():
        key = str(raw_key).strip().lower()
        if raw_value is None:
            value = None
        elif key in {'hessian', 'coulomb_method', 'inference', 'task', 'size'}:
            value = str(raw_value).strip().lower()
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
        implicit: str = 'None',
        solvent: str = 'None',
        model_options: Optional[dict] = None,
        solvation_options: Optional[dict] = None,
    ) -> None:
        self.output = output
        self.model = str(model).strip().lower()
        self.d4 = d4
        self.device = device
        self.atoms = atoms
        self.implicit = normalize_none_option(implicit)
        self.solvent = normalize_none_option(solvent)
        self.model_options = _normalize_model_options(model_options)
        self.solvation_options = solvation_options or {}
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

    def _validate_solvent_config(self) -> None:
        self.implicit, self.solvent = validate_implicit_solvent_choice(
            self.implicit, self.solvent
        )
        if self.implicit == 'none':
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

        normalized = name.lower()
        module_override = self.model_options.get('module')
        if module_override:
            import_calculator_plugin(str(module_override))
        else:
            builtin_name = _builtin_canonical_name(normalized)
            builtin_module = _BUILTIN_NAME_TO_MODULE.get(normalized)
            if builtin_module is None and builtin_name is not None:
                builtin_module = _BUILTIN_NAME_TO_MODULE.get(builtin_name)
            if builtin_module is not None:
                importlib.import_module(builtin_module)

        try:
            cls = get_registered_calculator(normalized)
            self.model = normalized
            return cls
        except KeyError:
            pass

        builtin_name = _builtin_canonical_name(normalized)
        if builtin_name is not None and builtin_name != normalized:
            builtin_module = _BUILTIN_NAME_TO_MODULE.get(builtin_name)
            if builtin_module is not None:
                importlib.import_module(builtin_module)
            try:
                cls = get_registered_calculator(builtin_name)
                self.model = builtin_name
                return cls
            except KeyError:
                pass

        raise ValueError(f"Unsupported model: '{name}'.")

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

        supported_coulomb = getattr(cls, 'SUPPORTED_COULOMB_METHODS', None)
        if supported_coulomb is not None:
            requested_coulomb = {
                key: str(self.model_options[key]).lower()
                for key in ('coulomb', 'coulomb_method')
                if self.model_options.get(key) is not None
            }
            if len(set(requested_coulomb.values())) > 1:
                raise ValueError(
                    f"Conflicting Coulomb options for '{self.model}': "
                    f"coulomb={self.model_options.get('coulomb')!r}, "
                    f"coulomb_method={self.model_options.get('coulomb_method')!r}. "
                    "Specify only one spelling or use matching values."
                )
            coulomb_method = next(iter(requested_coulomb.values()), None)
            if coulomb_method is not None and coulomb_method not in supported_coulomb:
                supported_text = ', '.join(sorted(supported_coulomb))
                if coulomb_method == 'ewald':
                    raise NotImplementedError(
                        "AIMNet2 Coulomb method 'ewald' requires validated PBC/cell/MIC support; "
                        f"use one of: {supported_text}."
                    )
                raise ValueError(
                    f"Model '{self.model}' does not support Coulomb method '{coulomb_method}'. "
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
        # Only annotate shipped models; registry plugins keep their own class
        # capability attributes instead of being force-overwritten to False.
        if self.model in MODEL_PBC_MD_SUPPORT:
            calculator.maple_pbc_md_supported = model_supports_pbc_md(self.model)
        if self.model in MODEL_STRESS_SUPPORT:
            calculator.maple_stress_supported = model_supports_stress(self.model)

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        requested_name = self.model
        self._validate_solvent_config()

        cls = self._discover_calculator_class(requested_name)
        name = self.model
        self._validate_model_options(cls)
        self._validate_against_class(cls)
        options = dict(self.model_options)
        option_keys = getattr(cls, 'OPTION_KEYS', None)
        if option_keys is not None and 'd4' in option_keys:
            options.setdefault('d4', self.d4)
        # Allow input header to override the auto-resolved model path.
        if options.get('model_path'):
            resolved_model_path = self._resolve_explicit_model_path(cls, options)
        else:
            resolved_model_path = self._resolve_model_path(cls, name)

        # UMA's `size`/`checkpoint_path` resolution is special: when no checkpoint
        # is given and the requested size has a known HF fallback, prefer a
        # locally-cached MAPLE copy if present. Handle the coercion here so that
        # UMACalculator only receives clean kwargs.
        if cls.__name__ == 'UMACalculator':
            from .uma._uma_calculator import UMACalculator, UMA_DEFAULT_SIZE, UMA_FALLBACK_HF_MODELS

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

        calculator = cls(
            device=self.device,
            model=name,
            implicit=self.implicit,
            solvent=self.solvent,
            **kwargs,
        )

        self._apply_hessian_mode(calculator)
        return calculator

    def _apply_hessian_mode(self, calculator) -> None:
        mode = self.model_options.get('hessian')
        if mode is None:
            return

        # Instance/MAPLE capability attr wins over the class protocol constant:
        # the official PBC adapters declare an empty ``supported_hessian_modes``
        # while inheriting CalcABC's non-empty default constant.
        supported = getattr(calculator, 'supported_hessian_modes', None)
        if supported is None:
            supported = type(calculator).SUPPORTED_HESSIAN_MODES
        if mode not in supported:
            if not supported:
                raise ValueError(f"Model '{self.model}' does not support Hessian modes.")
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
            self._annotate_calculator_capabilities(calculator)
            # General build path serves MD and non-MD (SP/OPT/SCAN/TS) tasks: keep the
            # known-cutoff minimum-image check, but do not block a calculator that does
            # not declare a cutoff here.  The strict unknown-cutoff rejection lives on
            # the MD admission path (validate_md_capabilities).
            validate_pbc_neighbor_cutoff(self.atoms, calculator, require_known_cutoff=False)
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

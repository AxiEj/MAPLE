from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..calculator_base import register_calculator


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_MACE_VERSION = "0.3.16"


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_mace_calculator():
    from mace.calculators import MACECalculator

    return MACECalculator


def _construct_upstream_calculator(
    *,
    kind: str,
    checkpoint: Path,
    device: str,
    default_dtype: str,
):
    try:
        version = importlib.metadata.version("mace-torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "MACE_RUNTIME_MISSING: install the pinned mace-torch runtime."
        ) from exc
    if version != _SUPPORTED_MACE_VERSION:
        raise RuntimeError(
            "MACE_RUNTIME_VERSION_MISMATCH: Route A requires mace-torch "
            f"{_SUPPORTED_MACE_VERSION}, found {version}."
        )

    try:
        calculator_class = _import_mace_calculator()
    except (ImportError, OSError) as exc:
        conda_lib = Path(sys.prefix) / "lib"
        raise RuntimeError(
            "RUNTIME_ABI_INCOMPATIBLE: mace-torch could not import its native "
            f"runtime ({type(exc).__name__}: {exc}). The active environment "
            f"library directory is '{conda_lib}', while LD_LIBRARY_PATH is "
            f"{os.environ.get('LD_LIBRARY_PATH', '<unset>')!r}. Prepend the "
            "active environment lib directory before launching MAPLE; the "
            "provider does not mutate process environment variables."
        ) from exc

    kwargs: dict[str, Any] = {
        "model_paths": str(checkpoint),
        "device": str(device),
        "default_dtype": default_dtype,
        "model_type": "MACE",
    }
    if kind == "omol":
        kwargs["head"] = "omol"
    elif kind != "off24":
        raise ValueError(f"Unknown upstream MACE provider kind: {kind}.")
    return calculator_class(**kwargs)


class _PinnedMACEProvider(Calculator):
    """Narrow eV-native adapter around upstream mace-torch.

    Route A uses this adapter inside ASE MD and free-energy estimators, where
    ASE's native eV/eV-per-Angstrom convention is required. It deliberately
    does not inherit MAPLE's Hartree-normalizing ``CalcABC``.
    """

    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES: tuple[str, ...] = ()
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ("checkpoint", "sha256", "license_ack")
    MODEL_PATH_OPTION = "checkpoint"
    DEFAULT_DTYPE = "float64"
    PROVIDER_KIND = ""
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    implemented_properties = ["energy", "free_energy", "forces"]

    @classmethod
    def build_kwargs_from_options(
        cls,
        model,
        options,
        *,
        resolved_model_path=None,
    ) -> dict[str, Any]:
        checkpoint = options.get("checkpoint")
        if resolved_model_path is not None:
            checkpoint = resolved_model_path
        return {
            "checkpoint": checkpoint,
            "sha256": options.get("sha256"),
            "license_ack": options.get("license_ack"),
        }

    def __init__(
        self,
        *,
        device,
        model: str,
        checkpoint: str,
        sha256: str,
        license_ack: bool,
        implicit: str = "none",
        solvent: str = "none",
    ) -> None:
        super().__init__()
        if str(implicit).lower() not in {"", "none"} or str(solvent).lower() not in {
            "",
            "none",
        }:
            raise ValueError(
                "Pinned upstream MACE providers are vacuum MLIPs; compose "
                "continuum response outside the provider."
            )
        if license_ack is not True:
            raise ValueError(
                "Pinned MACE foundation models require license_ack=true."
            )
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise ValueError(
                "Pinned upstream MACE checkpoint sha256 must be 64 lowercase "
                "hex characters."
            )
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(
                "Pinned upstream MACE requires an explicit checkpoint path."
            )
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ValueError(
                "MODEL_HASH_MISMATCH: pinned upstream MACE checkpoint does "
                f"not exist: {checkpoint_path}"
            )
        actual_sha256 = _raw_sha256(checkpoint_path)
        if actual_sha256 != sha256:
            raise ValueError(
                "MODEL_HASH_MISMATCH: pinned upstream MACE checkpoint "
                f"declares {sha256} but hashes to {actual_sha256}."
            )

        self.device = str(device)
        self.model_name = str(model)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = sha256
        self.license_acknowledged = True
        self.default_dtype = self.DEFAULT_DTYPE
        self._upstream = _construct_upstream_calculator(
            kind=self.PROVIDER_KIND,
            checkpoint=checkpoint_path,
            device=self.device,
            default_dtype=self.default_dtype,
        )

    def _prepare_atoms(self, atoms: Atoms) -> Atoms:
        raise NotImplementedError

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Pinned upstream MACE requires an Atoms object.")
        requested = tuple(properties or ("energy",))
        unsupported = sorted(set(requested) - set(self.implemented_properties))
        if unsupported:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide: "
                + ", ".join(unsupported)
            )

        probe = self._prepare_atoms(atoms)
        upstream_properties = [
            "energy" if name == "free_energy" else name for name in requested
        ]
        upstream_properties = list(dict.fromkeys(upstream_properties))
        self._upstream.calculate(
            probe,
            properties=upstream_properties,
            system_changes=system_changes,
        )
        source = self._upstream.results
        if "energy" not in source:
            raise RuntimeError(
                "MACE_RESULT_INVALID: upstream provider returned no energy."
            )

        energy = float(source["energy"])
        if not math.isfinite(energy):
            raise RuntimeError(
                "MACE_RESULT_INVALID: upstream provider returned non-finite energy."
            )
        results: dict[str, Any] = {
            "energy": energy,
            "free_energy": float(source.get("free_energy", energy)),
        }
        for name in ("forces", "stress"):
            if name not in requested:
                continue
            values = np.asarray(source.get(name), dtype=float)
            expected_shape = (len(atoms), 3) if name == "forces" else (6,)
            if values.shape != expected_shape or not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"MACE_RESULT_INVALID: upstream {name} must have finite "
                    f"shape {expected_shape}, found {values.shape}."
                )
            results[name] = values.copy()
        self.results = results

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "provider": type(self).__name__,
            "provider_kind": self.PROVIDER_KIND,
            "mace_torch_version": _SUPPORTED_MACE_VERSION,
            "checkpoint": self.checkpoint.as_posix(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "default_dtype": self.default_dtype,
            "device": self.device,
            "result_units": {
                "energy": "eV",
                "forces": "eV/angstrom",
                "stress": "eV/angstrom^3",
            },
        }


@register_calculator
class MACEOff24Provider(_PinnedMACEProvider):
    MODEL_NAMES = ("maceoff24m",)
    PROVIDER_KIND = "off24"
    SUPPORTS_PBC = True
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def _prepare_atoms(self, atoms: Atoms) -> Atoms:
        if not bool(np.all(atoms.get_pbc())):
            raise ValueError(
                "DOMAIN_UNSUPPORTED: MACE-OFF24 discovery requires three-dimensional PBC."
            )
        cell = np.asarray(atoms.cell.array, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError(
                "DOMAIN_UNSUPPORTED: MACE-OFF24 requires a finite 3x3 cell."
            )
        if abs(float(np.linalg.det(cell))) <= 1.0e-12:
            raise ValueError(
                "DOMAIN_UNSUPPORTED: MACE-OFF24 requires a non-singular cell."
            )
        return atoms.copy()


class MACEOMOLProvider(_PinnedMACEProvider):
    MODEL_NAMES = ()
    PROVIDER_KIND = "omol"
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False

    def _prepare_atoms(self, atoms: Atoms) -> Atoms:
        if bool(np.any(atoms.get_pbc())):
            raise ValueError(
                "DOMAIN_UNSUPPORTED: MACE-OMOL production states are nonperiodic."
            )
        charge = atoms.info.get("charge")
        multiplicity = atoms.info.get("mult")
        if charge != 0 or multiplicity != 1:
            raise ValueError(
                "DOMAIN_UNSUPPORTED: Route A MACE-OMOL v1 requires explicit "
                "charge=0 and mult=1."
            )
        existing_spin = atoms.info.get("spin")
        if existing_spin is not None and float(existing_spin) != 1.0:
            raise ValueError(
                "MACE-OMOL closed-shell metadata requires spin=1; spin=0 is "
                "not the multiplicity convention used by the upstream model."
            )
        probe = atoms.copy()
        probe.info["charge"] = 0.0
        probe.info["spin"] = 1.0
        return probe

"""Pinned highest-precision ASE adapter for public FeNNix-Bio1 checkpoints.

The adapter exposes only the pretrained explicit-system potential-energy
surface.  The paper's hydration and binding free energies require separate
alchemical sampling protocols and are not outputs of one calculator call.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from contextlib import AbstractContextManager
from importlib import metadata
from pathlib import Path, PurePath
from typing import Callable, Mapping, Protocol, Sequence, cast

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms

from ..calculator_base import CalcABC, register_calculator
from ..model_capabilities import ModelProvenanceCard

FENNIX_BIO1_SOURCE_REVISION = "d62b8740343b803a2b864140ec79e347f8ba034e"
FENNIX_BIO1_PMC_REVISION = "83f299b81c1d62e2a15c892280559a7c0cc2fac3"
FENNOL_DISTRIBUTION_VERSION = "2026.6.29"
FENNOL_ASE_SHA256 = "6d6a9dbf40fce96b8589a3702de7de190f8301d0827f629fbfad2c1e4a2f97d7"
FENNOL_PREPROCESSING_SHA256 = (
    "153519f469e112b6f1d2bfe7f243c241db476f273543abd0540ab658d59c96d4"
)
FENNIX_BIO1_CHECKPOINTS = {
    "small": {
        "filename": "fennix-bio1S.fnx",
        "sha256": "82c570c57e95cf164b1a1b0ac2122133cb435c89b07d495246773091541c2f07",
    },
    "medium": {
        "filename": "fennix-bio1M.fnx",
        "sha256": "5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a",
    },
}
FENNIX_BIO1_CARD = (
    Path(__file__).resolve().parents[1] / "model_cards" / "fennix-bio1.yaml"
)


class _CalculatorDelegate(Protocol):
    @property
    def results(self) -> Mapping[str, object]: ...

    def calculate(
        self,
        atoms: Atoms,
        properties: Sequence[str],
        system_changes: Sequence[str],
    ) -> None: ...


class _JAXModule(Protocol):
    def devices(self, backend: str | None = None) -> Sequence[object]: ...

    def default_device(self, device: object) -> AbstractContextManager[None]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_card(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"FeNNix-Bio1 model card must use JSON-compatible YAML syntax: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("FeNNix-Bio1 model card must be a JSON object.")
    return payload


def _validate_official_card(card: dict) -> ModelProvenanceCard:
    if str(card.get("model_id", "")).strip().lower() != "fennix-bio1":
        raise ValueError("FeNNix-Bio1 model card identity mismatch.")
    source = card.get("source")
    if (
        not isinstance(source, dict)
        or str(source.get("revision", "")).strip() != FENNIX_BIO1_SOURCE_REVISION
        or str(source.get("distribution_version", "")).strip()
        != FENNOL_DISTRIBUTION_VERSION
        or str(source.get("ase_adapter_sha256", "")).strip().lower()
        != FENNOL_ASE_SHA256
        or str(source.get("preprocessing_sha256", "")).strip().lower()
        != FENNOL_PREPROCESSING_SHA256
    ):
        raise ValueError(
            "FeNNix-Bio1 model card does not bind the pinned FeNNol runtime."
        )
    checkpoint_source = card.get("checkpoint_source")
    if (
        not isinstance(checkpoint_source, dict)
        or str(checkpoint_source.get("revision", "")).strip()
        != FENNIX_BIO1_PMC_REVISION
    ):
        raise ValueError(
            "FeNNix-Bio1 model card does not bind the pinned FeNNol-PMC revision."
        )
    checkpoints = card.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("FeNNix-Bio1 model card must define pinned checkpoints.")
    for size, expected in FENNIX_BIO1_CHECKPOINTS.items():
        record = checkpoints.get(size)
        if (
            not isinstance(record, dict)
            or record.get("filename") != expected["filename"]
            or str(record.get("checkpoint_sha256", "")).strip().lower()
            != expected["sha256"]
        ):
            raise ValueError(
                f"FeNNix-Bio1 model card {size!r} checkpoint identity mismatch."
            )

    provenance = ModelProvenanceCard.from_payload(card, model_name="fennix-bio1")
    capabilities = provenance.capabilities
    if (
        not capabilities.energy
        or not capabilities.forces
        or not capabilities.conservative_forces
        or capabilities.hessian != "finite_difference"
        or not capabilities.supports_md
        or not capabilities.supports_pbc
        or not capabilities.supports_multifragment
        or capabilities.solvation_mode != "none"
        or capabilities.supports_absolute_solvation
        or capabilities.supports_alchemical_lambda
    ):
        raise ValueError(
            "FeNNix-Bio1 card must describe a conservative explicit-system PES "
            "without an embedded free-energy protocol."
        )
    return provenance


def _official_calculator_factory(model_path: str) -> _CalculatorDelegate:
    try:
        fennol_ase = importlib.import_module("fennol.ase")
    except ImportError as exc:
        raise ImportError(
            "FeNNix-Bio1 requires the optional official FeNNol runtime."
        ) from exc
    calculator_type = getattr(fennol_ase, "FENNIXCalculator", None)
    if calculator_type is None:
        raise ImportError(
            "Installed FeNNol does not expose fennol.ase.FENNIXCalculator."
        )
    calculator = calculator_type(
        model=model_path,
        gpu_preprocessing=False,
        use_float64=True,
        matmul_prec="highest",
    )
    if not hasattr(calculator, "calculate") or not hasattr(calculator, "results"):
        raise TypeError(
            "FeNNol FENNIXCalculator does not satisfy the ASE delegate protocol."
        )
    return cast(_CalculatorDelegate, calculator)


def _validate_runtime_receipt(receipt: Mapping[str, object]) -> dict:
    if not isinstance(receipt, Mapping):
        raise ValueError("FeNNix-Bio1 runtime validator must return a mapping.")
    expected = {
        "distribution": "FeNNol",
        "version": FENNOL_DISTRIBUTION_VERSION,
        "ase_sha256": FENNOL_ASE_SHA256,
        "preprocessing_sha256": FENNOL_PREPROCESSING_SHA256,
    }
    normalized = {key: str(receipt.get(key, "")).strip() for key in expected}
    if normalized["distribution"].lower() != expected["distribution"].lower() or any(
        normalized[key] != expected[key]
        for key in ("version", "ase_sha256", "preprocessing_sha256")
    ):
        raise ValueError(
            "FeNNix-Bio1 requires the exact pinned FeNNol runtime "
            f"{FENNOL_DISTRIBUTION_VERSION} with audited source-file hashes."
        )
    return expected


def _official_runtime_validator() -> dict:
    try:
        distribution = metadata.distribution("FeNNol")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "FeNNix-Bio1 requires FeNNol distribution "
            f"{FENNOL_DISTRIBUTION_VERSION}."
        ) from exc
    ase_path = Path(str(distribution.locate_file(PurePath("fennol/ase.py"))))
    preprocessing_path = Path(
        str(distribution.locate_file(PurePath("fennol/models/preprocessing.py")))
    )
    if not ase_path.is_file() or not preprocessing_path.is_file():
        raise ImportError(
            "Installed FeNNol does not contain the audited ASE and preprocessing "
            "runtime files."
        )
    return _validate_runtime_receipt(
        {
            "distribution": str(distribution.metadata["Name"]),
            "version": distribution.version,
            "ase_sha256": _sha256(ase_path),
            "preprocessing_sha256": _sha256(preprocessing_path),
        }
    )


def _official_device_context_factory(device: str) -> AbstractContextManager:
    try:
        jax = cast(_JAXModule, importlib.import_module("jax"))
    except ImportError as exc:
        raise ImportError(
            "FeNNix-Bio1 requires JAX from the optional official FeNNol runtime."
        ) from exc

    requested = str(device).strip().lower()
    platform, separator, raw_index = requested.partition(":")
    if platform == "cuda":
        platform = "gpu"
    if platform not in {"cpu", "gpu"}:
        raise ValueError(
            "FeNNix-Bio1 device must be an explicit CPU or CUDA/GPU device."
        )
    index = 0
    if separator:
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(
                f"FeNNix-Bio1 device index must be an integer, got {device!r}."
            ) from exc
    devices = jax.devices(platform)
    if index < 0 or index >= len(devices):
        raise ValueError(
            f"FeNNix-Bio1 device {device!r} is unavailable; "
            f"JAX reports {len(devices)} {platform} device(s)."
        )
    return jax.default_device(devices[index])


def _normalized_model_and_size(model: str, size: str | None) -> str:
    requested = (
        str(model).strip().lower().replace("-", "").replace("_", "").replace(".", "")
    )
    aliases = {
        "fennixbio1": None,
        "fennixbio1small": "small",
        "fennixbio1s": "small",
        "fennixbio1medium": "medium",
        "fennixbio1m": "medium",
    }
    if requested not in aliases:
        raise ValueError(
            "FeNNix-Bio1 is pinned to the 'fennix-bio1' model family; "
            f"received {model!r}."
        )
    alias_size = aliases[requested]
    selected = str(size).strip().lower() if size is not None else alias_size or "small"
    if selected in {"s", "bio1s"}:
        selected = "small"
    elif selected in {"m", "bio1m"}:
        selected = "medium"
    if selected not in FENNIX_BIO1_CHECKPOINTS:
        raise ValueError("FeNNix-Bio1 size must be 'small' or 'medium'.")
    if alias_size is not None and selected != alias_size:
        raise ValueError(f"FeNNix-Bio1 alias {model!r} conflicts with size={size!r}.")
    return selected


@register_calculator
class FeNNixBio1Calculator(CalcABC):
    """Official FeNNix-Bio1 explicit-system potential with fail-closed precision."""

    implemented_properties = ["energy", "forces", "free_energy", "hessian"]
    MODEL_NAMES = (
        "fennix-bio1",
        "fennix-bio1-small",
        "fennix-bio1-medium",
    )
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = True
    OPTION_KEYS = ("size",)
    MODEL_PATH_OPTION = "model_path"

    @classmethod
    def build_kwargs_from_options(
        cls,
        model,
        model_options,
        *,
        resolved_model_path=None,
    ):
        if model_options.get("d4", False):
            raise ValueError(
                "FeNNix-Bio1 does not support D4 composition; refusing to ignore it."
            )
        kwargs = {"size": model_options.get("size")}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device,
        model: str = "fennix-bio1",
        model_path: str | None = None,
        implicit: str = "none",
        solvent: str = "none",
        size: str | None = None,
        model_card_path: str | None = None,
        calculator_factory: Callable[[str], _CalculatorDelegate] | None = None,
        device_context_factory: Callable[[str], AbstractContextManager] | None = None,
        runtime_validator: Callable[[], Mapping[str, object]] | None = None,
        execution_task: str | None = None,
    ) -> None:
        super().__init__()
        selected_size = _normalized_model_and_size(model, size)
        implicit_name = str(implicit).strip().lower()
        solvent_name = str(solvent).strip().lower()
        if implicit_name not in {"", "none"} or solvent_name not in {"", "none"}:
            raise ValueError(
                "FeNNix-Bio1 is an explicit-system PES and cannot be composed "
                "with an additive implicit backend or named implicit solvent."
            )
        if (
            model_card_path is not None
            and Path(model_card_path).expanduser().resolve()
            != FENNIX_BIO1_CARD.resolve()
        ):
            raise ValueError(
                "FeNNix-Bio1 binds a fixed official model card and does not "
                "accept a user-supplied model_card_path."
            )

        card = _load_card(FENNIX_BIO1_CARD)
        self.model_provenance = _validate_official_card(card)
        self.model_provenance.validate_task(execution_task or "sp")
        self.model_provenance.validate_device(
            device,
            task=execution_task or "sp",
        )
        if model_path is None:
            raise ValueError(
                "FeNNix-Bio1 requires an explicit model_path to an official "
                "fennix-bio1S.fnx or fennix-bio1M.fnx checkpoint."
            )
        checkpoint = Path(model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FeNNix-Bio1 checkpoint not found: {checkpoint}")
        expected = FENNIX_BIO1_CHECKPOINTS[selected_size]["sha256"]
        actual = _sha256(checkpoint)
        if actual != expected:
            other_sizes = [
                other
                for other, record in FENNIX_BIO1_CHECKPOINTS.items()
                if record["sha256"] == actual
            ]
            if other_sizes:
                raise ValueError(
                    f"FeNNix-Bio1 checkpoint is {other_sizes[0]!r}, not the "
                    f"requested {selected_size!r} variant."
                )
            raise ValueError(
                "FeNNix-Bio1 checkpoint checksum mismatch: "
                f"expected {expected}, got {actual}."
            )
        self.runtime_receipt = _validate_runtime_receipt(
            (runtime_validator or _official_runtime_validator)()
        )

        self.device = str(device)
        self.model_name = "fennix-bio1"
        self.model_size = selected_size
        self.model_path = str(checkpoint)
        self.model_card_path = str(FENNIX_BIO1_CARD)
        self.model_card = card
        self.execution_task = execution_task
        self.capabilities = self.model_provenance.capabilities
        self.hessian = "numerical"
        self.solvent_correction = None
        self._device_context_factory = (
            device_context_factory or _official_device_context_factory
        )
        factory = calculator_factory or _official_calculator_factory
        with self._device_context_factory(self.device):
            self._delegate = factory(self.model_path)

    @property
    def provenance(self) -> dict:
        checkpoint = self.model_card["checkpoints"][self.model_size]
        return {
            "provider": "fennix-bio1",
            "variant": self.model_size,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_path": self.model_path,
            "source_revision": self.model_card["source"]["revision"],
            "runtime_receipt": dict(self.runtime_receipt),
            "checkpoint_source_revision": self.model_card["checkpoint_source"][
                "revision"
            ],
            "model_card": self.model_card_path,
            "precision": {
                "requested_float64": True,
                "cpu_neighborlist_coordinate_dtype": "float32",
                "effective_precision": "unverified_mixed_or_checkpoint_serialized",
                "end_to_end_float64_verified": False,
                "matmul_precision": "highest",
                "gpu_preprocessing": False,
            },
            "runtime_scope": "explicit_system_pes_only",
        }

    def validate_task(self, task: str) -> None:
        self.model_provenance.validate_task(task)

    def get_property(self, name, atoms=None, allow_calculation=True):
        """Validate metadata even when ASE can serve a cached result."""

        target_atoms = atoms if atoms is not None else getattr(self, "atoms", None)
        if target_atoms is not None:
            self._validate_system(target_atoms)
        return super().get_property(
            name,
            atoms=atoms,
            allow_calculation=allow_calculation,
        )

    def _validate_system(self, atoms) -> None:
        if len(atoms) == 0:
            raise ValueError("FeNNix-Bio1 requires at least one atom.")
        if "charge" not in atoms.info or "mult" not in atoms.info:
            raise ValueError(
                "FeNNix-Bio1 requires explicit atoms.info['charge'] and "
                "atoms.info['mult']; MAPLE will not assume neutral singlet metadata."
            )
        try:
            charge = float(atoms.info["charge"])
            multiplicity = float(atoms.info["mult"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "FeNNix-Bio1 requires finite integer charge and multiplicity."
            ) from exc
        if (
            not math.isfinite(charge)
            or not charge.is_integer()
            or not math.isfinite(multiplicity)
            or not multiplicity.is_integer()
        ):
            raise ValueError(
                "FeNNix-Bio1 requires finite integer charge and multiplicity."
            )
        if charge != 0.0 or multiplicity != 1.0:
            raise ValueError(
                "FeNNix-Bio1 is initially admitted only for explicit neutral singlets."
            )
        electron_count = int(
            np.asarray(atoms.get_atomic_numbers(), dtype=int).sum()
        ) - int(charge)
        if multiplicity == 1.0 and electron_count % 2:
            raise ValueError(
                "FeNNix-Bio1 mult=1 requires an even electron count; the declared "
                "neutral singlet metadata is inconsistent with this composition."
            )
        try:
            initial_charges = np.asarray(atoms.get_initial_charges(), dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "FeNNix-Bio1 requires a finite initial-charge vector."
            ) from exc
        if (
            initial_charges.shape != (len(atoms),)
            or not np.isfinite(initial_charges).all()
        ):
            raise ValueError(
                "FeNNix-Bio1 requires a finite initial-charge vector with one "
                "value per atom."
            )
        initial_charge_sum = float(initial_charges.sum())
        if not math.isclose(initial_charge_sum, charge, abs_tol=1.0e-8, rel_tol=0.0):
            raise ValueError(
                "FeNNix-Bio1 initial-charge sum must match the declared "
                f"atoms.info['charge']; got {initial_charge_sum} and {charge}."
            )
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 86):
            raise ValueError(
                "FeNNix-Bio1 can only represent atomic numbers 1 through 86 "
                "according to the pinned configuration; this boundary is not a "
                "validated chemical-accuracy domain."
            )

        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        if pbc.any() and not pbc.all():
            raise ValueError(
                "FeNNix-Bio1 accepts either a fully periodic or non-periodic system."
            )
        if pbc.all():
            cell = np.asarray(atoms.cell.array, dtype=float)
            if (
                cell.shape != (3, 3)
                or not np.isfinite(cell).all()
                or not math.isfinite(float(atoms.cell.volume))
                or float(atoms.cell.volume) <= 0.0
            ):
                raise ValueError(
                    "Periodic FeNNix-Bio1 systems require a finite positive-volume cell."
                )

    def _validate_properties(self, properties) -> tuple[str, ...]:
        normalized = tuple(str(item).strip().lower() for item in properties)
        unsupported = sorted(
            set(normalized).difference({"energy", "free_energy", "forces", "hessian"})
        )
        if unsupported:
            raise ValueError(
                "FeNNix-Bio1 received unsupported properties: " + ", ".join(unsupported)
            )
        self.validate_task("sp")
        if "forces" in normalized:
            self.validate_task("opt")
        if "hessian" in normalized:
            self.validate_task("frequency")
        return normalized

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        properties = self._validate_properties(self._normalize_properties(properties))
        calculated_atoms = super().calculate(atoms, properties, system_changes)
        if calculated_atoms is None:
            raise ValueError("FeNNix-Bio1 calculation requires an ASE Atoms object.")
        atoms = calculated_atoms
        self._validate_system(atoms)

        delegated = ["energy"]
        if "forces" in properties or "hessian" in properties:
            delegated.append("forces")
        with self._device_context_factory(self.device):
            self._delegate.calculate(
                atoms,
                properties=delegated,
                system_changes=all_changes,
            )
        raw = getattr(self._delegate, "results", {})
        if "energy" not in raw:
            raise RuntimeError("FeNNix-Bio1 delegate did not return energy.")
        try:
            energy = float(raw["energy"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("FeNNix-Bio1 delegate energy must be finite.") from exc
        if not math.isfinite(energy):
            raise ValueError("FeNNix-Bio1 delegate energy must be finite.")

        forces = None
        if "forces" in properties or "hessian" in properties:
            if "forces" not in raw:
                raise RuntimeError("FeNNix-Bio1 delegate did not return forces.")
            forces = np.asarray(raw["forces"], dtype=float)
            if forces.shape != (len(atoms), 3):
                raise ValueError("FeNNix-Bio1 delegate forces must have shape (N, 3).")
            if not np.isfinite(forces).all():
                raise ValueError("FeNNix-Bio1 delegate forces must be finite.")

        hessian = self.get_hessian(atoms) if "hessian" in properties else None
        self._finalize_results(
            atoms,
            energy=energy,
            forces=forces,
            hessian=hessian,
        )
        self.results["model_metadata"] = self.provenance

    def _validate_conservative_forces(
        self,
        atoms,
        *,
        delta: float = 1.0e-4,
        absolute_tolerance: float = 2.0e-5,
        relative_tolerance: float = 2.0e-4,
    ) -> None:
        """Verify ``F=-dE/dR`` in MAPLE units before numerical frequencies."""

        old_results = dict(getattr(self, "results", {}) or {})
        old_atoms = getattr(self, "atoms", None)
        fixed = {
            index
            for constraint in getattr(atoms, "constraints", []) or []
            if isinstance(constraint, FixAtoms)
            for index in constraint.get_indices()
        }
        try:
            reference = atoms.copy()
            reference.set_constraint()
            self.calculate(reference, properties=("energy", "forces"))
            analytic = np.asarray(self.results["forces"], dtype=float)
            numeric = np.zeros_like(analytic)
            positions = reference.get_positions().copy()
            for atom_index in range(len(reference)):
                if atom_index in fixed:
                    continue
                for axis in range(3):
                    plus = reference.copy()
                    plus_positions = positions.copy()
                    plus_positions[atom_index, axis] += delta
                    plus.set_positions(plus_positions)
                    self.calculate(plus, properties=("energy",))
                    energy_plus = float(self.results["energy"])

                    minus = reference.copy()
                    minus_positions = positions.copy()
                    minus_positions[atom_index, axis] -= delta
                    minus.set_positions(minus_positions)
                    self.calculate(minus, properties=("energy",))
                    energy_minus = float(self.results["energy"])
                    numeric[atom_index, axis] = -(energy_plus - energy_minus) / (
                        2.0 * delta
                    )

            movable = [index for index in range(len(reference)) if index not in fixed]
            if movable and not np.allclose(
                analytic[movable],
                numeric[movable],
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ):
                maximum_error = float(
                    np.max(np.abs(analytic[movable] - numeric[movable]))
                )
                raise ValueError(
                    "FeNNix-Bio1 numerical Hessian is disabled because the "
                    "runtime force/energy consistency check failed "
                    f"(maximum error {maximum_error:.6g} Hartree/angstrom)."
                )
        finally:
            self.results = old_results
            self.atoms = old_atoms

    def get_hessian(self, atoms, delta: float = 0.002):
        self.validate_task("frequency")
        self._validate_system(atoms)
        self._validate_conservative_forces(atoms)
        return super().get_hessian(atoms, delta=delta)

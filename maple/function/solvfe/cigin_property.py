"""Pinned, property-only adapter for the official CIGIN predictor.

CIGIN accepts one solute SMILES and one solvent SMILES and predicts a scalar
solvation property.  It is intentionally not an ASE calculator: the upstream
model exposes neither a potential-energy surface nor forces.  The original
runtime is forced to CPU because its CUDA branch allocates an interaction map
on CPU while the learned tensors are on CUDA; MAPLE does not patch that source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from rdkit import Chem

CIGIN_SOURCE_REVISION = "9990f826f1a17590f1925dfd470423633275840c"
CIGIN_SOURCE_URL = "https://github.com/devalab/CIGIN"
CIGIN_CHECKPOINT_SHA256 = (
    "79f07c6475f8adfce6913b61a7e1e911d5a049f031954aeeb1eae986d2ce4cce"
)
CIGIN_SUPPORTED_ELEMENTS = frozenset(
    {"C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"}
)


class CIGINConfigError(ValueError):
    """Raised when the pinned CIGIN source or requested domain is invalid."""


class CIGINRuntimeError(RuntimeError):
    """Raised when the upstream CIGIN prediction process cannot complete."""


@dataclass(frozen=True)
class CIGINPropertyResult:
    """A direct CIGIN scalar prediction, not a MAPLE energy calculation."""

    predicted_solvation_free_energy_kcal_mol: float
    solute_smiles: str
    solvent_smiles: str
    source_revision: str = CIGIN_SOURCE_REVISION
    checkpoint_sha256: str = CIGIN_CHECKPOINT_SHA256
    unit: str = "kcal/mol"
    target_quantity: str = "property_prediction"
    property_only: bool = True
    absolute_solvation_backend: bool = False
    execution_device: str = "cpu"
    cuda_disabled_for_upstream_device_consistency: bool = True


_PREDICT_SCRIPT = r"""
import json
import math
import sys
from pathlib import Path

import torch

source_root = Path(sys.argv[1]).resolve()
solute_smiles = sys.argv[2]
solvent_smiles = sys.argv[3]
scripts_directory = source_root / "scripts"
if str(scripts_directory) not in sys.path:
    sys.path.insert(0, str(scripts_directory))
from models import Cigin

if torch.cuda.is_available():
    raise RuntimeError("CIGIN child must have CUDA disabled before importing upstream modules.")

model = Cigin().to("cpu")
state_dict = torch.load(
    source_root / "weights" / "cigin.tar", map_location="cpu", weights_only=True
)
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    prediction, interaction_map = model(solute_smiles, solvent_smiles)
if prediction.shape != (1, 1) or interaction_map.ndim != 2:
    raise RuntimeError("CIGIN upstream predictor returned an unexpected tensor shape.")
value = float(prediction.item())
if not math.isfinite(value):
    raise RuntimeError("CIGIN upstream predictor returned a non-finite prediction.")
print(json.dumps({
    "prediction_kcal_mol": value,
    "solute_smiles": solute_smiles,
    "solvent_smiles": solvent_smiles,
    "execution_device": "cpu",
}, sort_keys=True))
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_origin(url: str) -> str:
    normalized = str(url).strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    lowered = normalized.casefold()
    if lowered.startswith("git@github.com:"):
        return "https://github.com/" + normalized.split(":", 1)[1].casefold()
    if lowered.startswith("ssh://git@github.com/"):
        return "https://github.com/" + normalized.split("github.com/", 1)[1].casefold()
    return lowered


def _git_result(source_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise CIGINConfigError(
            "CIGIN source identity verification requires git."
        ) from exc


def _git_output(source_root: Path, *args: str) -> str:
    result = _git_result(source_root, *args)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CIGINConfigError(
            f"Unable to verify the CIGIN source checkout ({' '.join(args)}): {detail}"
        )
    return result.stdout.strip()


def _verify_pinned_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise CIGINConfigError(f"Pinned CIGIN {label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise CIGINConfigError(
            f"Pinned CIGIN {label} SHA256 mismatch: expected {expected_sha256}, got {observed}."
        )


def _canonical_neutral_smiles(value: str, role: str) -> str:
    smiles = str(value).strip()
    if not smiles:
        raise CIGINConfigError(f"CIGIN {role} SMILES must be non-empty.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise CIGINConfigError(f"CIGIN {role} SMILES is invalid.")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise CIGINConfigError(f"CIGIN does not accept disconnected {role} fragments.")
    if Chem.GetFormalCharge(molecule) != 0:
        raise CIGINConfigError(f"CIGIN is restricted to neutral {role} molecules.")
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise CIGINConfigError(f"CIGIN does not accept radical {role} molecules.")
    unsupported = sorted(
        {atom.GetSymbol() for atom in molecule.GetAtoms()} - CIGIN_SUPPORTED_ELEMENTS
    )
    if unsupported:
        raise CIGINConfigError(
            "CIGIN does not support " f"{role} element(s): {', '.join(unsupported)}."
        )
    explicit_hydrogen_molecule = Chem.AddHs(molecule)
    if any(atom.GetDegree() > 4 for atom in explicit_hydrogen_molecule.GetAtoms()):
        raise CIGINConfigError(
            f"CIGIN upstream atom features support at most degree four in {role} molecules."
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


class CIGINPropertyAdapter:
    """Execute exact CIGIN scalar inference in an isolated CPU subprocess.

    ``source_root`` must be a clean official checkout at
    :data:`CIGIN_SOURCE_REVISION`.  The source tree and checkpoint remain
    upstream-owned; MAPLE only launches them with CUDA disabled.
    """

    def __init__(self, source_root: str | Path, *, timeout_seconds: float = 120.0):
        root = Path(source_root).expanduser().resolve()
        if not root.is_dir():
            raise CIGINConfigError(f"CIGIN source_root is not a directory: {root}")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise CIGINConfigError("timeout_seconds must be finite and positive.")
        self.source_root = root
        self.timeout_seconds = float(timeout_seconds)
        self.assert_source_identity()

    def assert_source_identity(self) -> None:
        """Reject source checkouts or checkpoint files outside the pinned identity."""

        if (
            _git_output(self.source_root, "rev-parse", "--is-inside-work-tree")
            != "true"
        ):
            raise CIGINConfigError("CIGIN source_root must be a git work tree.")
        revision = _git_output(self.source_root, "rev-parse", "HEAD")
        if revision != CIGIN_SOURCE_REVISION:
            raise CIGINConfigError(
                "CIGIN source revision mismatch: "
                f"expected {CIGIN_SOURCE_REVISION}, got {revision}."
            )
        origin = _normalize_origin(
            _git_output(self.source_root, "config", "--get", "remote.origin.url")
        )
        if origin != _normalize_origin(CIGIN_SOURCE_URL):
            raise CIGINConfigError(
                f"CIGIN source must have official origin {CIGIN_SOURCE_URL}, got {origin!r}."
            )
        if _git_output(
            self.source_root, "status", "--porcelain", "--untracked-files=all"
        ):
            raise CIGINConfigError(
                "CIGIN source checkout has tracked or untracked modifications; "
                "restore the pinned source."
            )
        for diff_args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            result = _git_result(self.source_root, *diff_args)
            if result.returncode == 1:
                raise CIGINConfigError(
                    "CIGIN source checkout has tracked modifications; restore the pinned source."
                )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise CIGINConfigError(
                    f"Unable to verify the CIGIN source checkout ({' '.join(diff_args)}): {detail}"
                )
        _verify_pinned_file(
            self.source_root / "weights" / "cigin.tar",
            CIGIN_CHECKPOINT_SHA256,
            "prediction checkpoint",
        )

    def predict(self, solute_smiles: str, solvent_smiles: str) -> CIGINPropertyResult:
        """Predict one neutral, single-fragment solute/solvent SMILES pair.

        This syntactic solvent input is not evidence of a verified multi-solvent
        training set and must not be interpreted as a solvation free-energy PES.
        """

        canonical_solute = _canonical_neutral_smiles(solute_smiles, "solute")
        canonical_solvent = _canonical_neutral_smiles(solvent_smiles, "solvent")
        self.assert_source_identity()
        payload = self._run_script(
            _PREDICT_SCRIPT, str(self.source_root), canonical_solute, canonical_solvent
        )
        try:
            prediction = float(payload["prediction_kcal_mol"])
            returned_solute = str(payload["solute_smiles"])
            returned_solvent = str(payload["solvent_smiles"])
            execution_device = str(payload["execution_device"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CIGINRuntimeError(
                "CIGIN upstream predictor returned an invalid result payload."
            ) from exc
        if not math.isfinite(prediction):
            raise CIGINRuntimeError(
                "CIGIN upstream predictor returned a non-finite prediction."
            )
        if (
            returned_solute != canonical_solute
            or returned_solvent != canonical_solvent
            or execution_device != "cpu"
        ):
            raise CIGINRuntimeError(
                "CIGIN upstream predictor returned a mismatched identity or device."
            )
        return CIGINPropertyResult(
            predicted_solvation_free_energy_kcal_mol=prediction,
            solute_smiles=canonical_solute,
            solvent_smiles=canonical_solvent,
        )

    def _child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        scripts_directory = str(self.source_root / "scripts")
        inherited_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = scripts_directory
        if inherited_path:
            environment["PYTHONPATH"] += os.pathsep + inherited_path
        interpreter_lib_directory = str(Path(sys.prefix) / "lib")
        inherited_library_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = interpreter_lib_directory
        if inherited_library_path:
            environment["LD_LIBRARY_PATH"] += os.pathsep + inherited_library_path
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    def _run_script(self, script: str, *arguments: str) -> Mapping[str, object]:
        try:
            result = subprocess.run(
                [sys.executable, "-c", script, *arguments],
                cwd=self.source_root,
                env=self._child_environment(),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CIGINRuntimeError(
                f"CIGIN upstream predictor exceeded its {self.timeout_seconds:g}-second timeout."
            ) from exc
        except OSError as exc:
            raise CIGINRuntimeError(
                "Unable to start the CIGIN upstream predictor."
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise CIGINRuntimeError(f"CIGIN upstream predictor failed: {detail}")
        for line in reversed(result.stdout.splitlines()):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                return payload
        raise CIGINRuntimeError(
            "CIGIN upstream predictor did not emit a JSON result payload."
        )

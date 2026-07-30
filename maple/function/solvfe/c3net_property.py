"""Pinned, property-only adapter for the upstream C3Net predictor.

C3Net predicts a scalar solvation property from a supplied SDF geometry and an
upstream solvent identifier.  It is deliberately isolated from the ASE
calculator path: it exposes neither a potential-energy surface nor forces.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

C3NET_SOURCE_REVISION = "191d5a928fb4787d8a6dfb9d0851ee3f74b074d0"
C3NET_SOURCE_URL = "https://github.com/SehanLee/C3Net"
C3NET_CHECKPOINT_SHA256 = (
    "9d018d4af7f4c2626f6d9524a9d6cc027083004c6532d1db713fe121eb9dcfab"
)
C3NET_EMBEDDING_SHA256 = (
    "6db6c78271032851b4073f792d50f41d27fa1c8ffb6e90218955c6246aa0f8b4"
)
C3NET_SUPPORTED_ELEMENTS = frozenset(
    {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
)
C3NET_SOLVENT_COUNT = 103


class C3NetConfigError(ValueError):
    """Raised when the pinned C3Net source or requested domain is invalid."""


class C3NetRuntimeError(RuntimeError):
    """Raised when the upstream C3Net prediction process cannot complete."""


@dataclass(frozen=True)
class C3NetPropertyResult:
    """A direct C3Net scalar prediction, not a MAPLE energy calculation."""

    predicted_solvation_free_energy_kcal_mol: float
    solvent: str
    solvent_id: int
    source_revision: str = C3NET_SOURCE_REVISION
    checkpoint_sha256: str = C3NET_CHECKPOINT_SHA256
    embedding_sha256: str = C3NET_EMBEDDING_SHA256
    unit: str = "kcal/mol"
    target_quantity: str = "property_prediction"
    conformer_policy: str = "one supplied SDF record; upstream n_conformer=1"
    conformer_count: int = 1
    component_predictions_kcal_mol: tuple[float, ...] = ()
    property_only: bool = True
    absolute_solvation_backend: bool = False
    numpy_int_compatibility_shim: bool = False


_PREDICT_SCRIPT = r"""
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

numpy_int_compatibility_shim = False
if not hasattr(np, "int"):
    # C3Net's 2023 dataset loader uses NumPy's removed alias.  On this
    # supported runtime, the historical np.int value is Python's native int.
    np.int = int
    numpy_int_compatibility_shim = True

import torch
from rdkit import Chem
from module.data import MoleculeLoader, MoleculeSet
from module.input import element
from module.input.input_generator import generator_predict
from module.nn import Atomistic_Model, Atomwise, C3Net
from prediction.predictor import Predictor

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision("highest")

sdf_path = Path(sys.argv[1]).resolve()
solvent = sys.argv[2]
conformer_mode = sys.argv[3]
solvent_keys = tuple(element.solv_prop_nomalized)
property_keys = {"logp", "pampa"}
solvents = tuple(key for key in solvent_keys if key not in property_keys)
if len(solvent_keys) != 105 or len(solvents) != 103 or "water" not in solvents:
    raise RuntimeError("Pinned C3Net solvent registry does not have its expected 103 solvent entries.")
if solvent not in solvents:
    raise ValueError("Unsupported C3Net solvent identifier: %r" % solvent)


def v2000_declared_stereo_identity(mol_block, molecule):
    lines = mol_block.splitlines()
    counts_index = next(
        (index for index, line in enumerate(lines) if "V2000" in line),
        None,
    )
    if counts_index is None:
        raise ValueError(
            "C3Net multi-conformer prediction requires auditable V2000 "
            "stereochemistry."
        )
    counts = lines[counts_index].split()
    atom_count, bond_count = int(counts[0]), int(counts[1])
    if (
        atom_count != molecule.GetNumAtoms()
        or counts_index + 1 + atom_count + bond_count > len(lines)
    ):
        raise ValueError("C3Net V2000 counts do not match the parsed molecule.")
    declared_tetrahedral_atoms = set()
    for atom_index, line in enumerate(
        lines[counts_index + 1 : counts_index + 1 + atom_count]
    ):
        parity = int(line.split()[6])
        if parity in {1, 2}:
            declared_tetrahedral_atoms.add(atom_index)
    unknown_double_bonds = []
    for line in lines[
        counts_index + 1 + atom_count : counts_index + 1 + atom_count + bond_count
    ]:
        begin, end, bond_type, stereo_code = map(int, line.split()[:4])
        if bond_type == 1 and stereo_code in {1, 6}:
            declared_tetrahedral_atoms.add(begin - 1)
        if bond_type == 2 and stereo_code == 3:
            unknown_double_bonds.append((begin - 1, end - 1))
    declared = Chem.Mol(molecule)
    for atom in declared.GetAtoms():
        if atom.GetIdx() not in declared_tetrahedral_atoms:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for begin, end in unknown_double_bonds:
        bond = declared.GetBondBetweenAtoms(begin, end)
        if bond is None:
            raise ValueError("C3Net V2000 bond is absent from the parsed molecule.")
        bond.SetStereo(Chem.BondStereo.STEREOANY)
        for atom_index in (begin, end):
            for neighboring_bond in declared.GetAtomWithIdx(atom_index).GetBonds():
                if neighboring_bond.GetIdx() != bond.GetIdx():
                    neighboring_bond.SetBondDir(Chem.BondDir.NONE)
    if declared.GetNumConformers():
        declared.GetConformer().Set3D(False)
    declared = Chem.RemoveHs(declared, sanitize=True)
    identity = Chem.MolToSmiles(
        declared,
        canonical=True,
        isomericSmiles=True,
    )
    if not identity:
        raise ValueError("C3Net SDF record lacks a declared-stereo identity.")
    return identity


sdf_blocks = [
    block.strip()
    for block in sdf_path.read_text(encoding="utf-8").split("$$$$")
    if block.strip()
]
molecules = list(Chem.SDMolSupplier(str(sdf_path), True, False))
if len(sdf_blocks) != len(molecules):
    raise ValueError("C3Net SDF record boundaries do not match parsed molecules.")
if conformer_mode == "single":
    if len(molecules) != 1:
        raise ValueError("C3Net requires exactly one valid molecule record in the supplied SDF.")
elif conformer_mode == "up_to_five":
    if not 1 <= len(molecules) <= 5:
        raise ValueError("C3Net multi-conformer prediction requires one to five SDF records.")
else:
    raise ValueError("Unknown C3Net conformer mode: %r" % conformer_mode)
if any(molecule is None for molecule in molecules):
    raise ValueError("C3Net requires valid molecule records in the supplied SDF.")

solute_identity = None
declared_stereo_identity = None
for mol_block, molecule in zip(sdf_blocks, molecules):
    if molecule.GetNumConformers() != 1:
        raise ValueError("Each C3Net SDF record requires exactly one supplied three-dimensional conformer.")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("C3Net does not accept disconnected solute fragments.")
    if Chem.GetFormalCharge(molecule) != 0:
        raise ValueError("C3Net is restricted to neutral solutes.")
    unsupported = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()} - {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
    if unsupported:
        raise ValueError("C3Net does not support element(s): %s" % ", ".join(unsupported))
    canonical_solute = Chem.MolToSmiles(
        Chem.RemoveHs(Chem.Mol(molecule), sanitize=True),
        canonical=True,
        isomericSmiles=False,
    )
    declared_stereo = v2000_declared_stereo_identity(mol_block, molecule)
    if solute_identity is None:
        solute_identity = canonical_solute
        declared_stereo_identity = declared_stereo
    elif (
        canonical_solute != solute_identity
        or declared_stereo != declared_stereo_identity
    ):
        raise ValueError("C3Net multi-conformer prediction requires records of one solute only.")

with tempfile.TemporaryDirectory(prefix="maple-c3net-") as temporary_directory:
    temporary_directory = Path(temporary_directory)
    npz_dir = temporary_directory / "npz"
    output_dir = temporary_directory / "output"
    npz_dir.mkdir()
    output_dir.mkdir()

    generator_predict(solvent, str(sdf_path), str(npz_dir))
    dataset = MoleculeSet(str(npz_dir), str(output_dir), 1)
    if len(dataset) != len(molecules):
        raise RuntimeError("C3Net upstream input generation did not preserve every supplied conformer.")
    dataset.npzs = sorted(
        dataset.npzs,
        key=lambda value: tuple(int(part) for part in Path(value).stem.split("_")),
    )
    loader = MoleculeLoader(dataset, batch_size=2, num_workers=0)
    model = Atomistic_Model(C3Net(), Atomwise(n_in=64))
    predictor = Predictor(str(output_dir), model, loader, "cpu")
    model.eval()
    predictions = []
    solute_ids = set()
    with torch.inference_mode():
        for data in loader:
            batch = {key: value.to(torch.device("cpu")) for key, value in data.items()}
            output = model(batch)
            values = output["y"].detach().cpu()
            for index in range(len(values)):
                solvent_id = int(batch["Solvent_ID"][index].reshape(-1)[0].item())
                solute_id = int(batch["Solute_ID"][index].reshape(-1)[0].item())
                prediction = float(values[index].reshape(-1)[0].item())
                if solvent_id != solvent_keys.index(solvent) or not math.isfinite(prediction):
                    raise RuntimeError("C3Net upstream prediction result failed its identity or finite-value check.")
                solute_ids.add(solute_id)
                predictions.append(prediction)
    # generator_predict writes Conf_ID=0 and composes IDs as ``mol_index``
    # followed by ``Conf_ID``; for its at-most-five-record convention this is
    # 0, 10, 20, 30, 40 rather than consecutive integers.
    if solute_ids != {index * 10 for index in range(len(molecules))}:
        raise RuntimeError("C3Net upstream prediction result did not preserve conformer identities.")
    dataset.clean_npzs()

print(json.dumps({
    "predictions_kcal_mol": predictions,
    "solvent": solvent,
    "solvent_id": solvent_id,
    "numpy_int_compatibility_shim": numpy_int_compatibility_shim,
}, sort_keys=True))
"""


_SOLVENTS_SCRIPT = r"""
import json
from module.input import element

property_keys = {"logp", "pampa"}
solvents = tuple(key for key in element.solv_prop_nomalized if key not in property_keys)
print(json.dumps({"solvents": solvents}, sort_keys=True))
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
    git_executable = Path("/usr/bin/git")
    command = str(git_executable) if git_executable.is_file() else "git"
    try:
        return subprocess.run(
            [command, "-C", str(source_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise C3NetConfigError(
            "C3Net source identity verification requires git."
        ) from exc


def _git_output(source_root: Path, *args: str) -> str:
    result = _git_result(source_root, *args)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise C3NetConfigError(
            f"Unable to verify the C3Net source checkout ({' '.join(args)}): {detail}"
        )
    return result.stdout.strip()


def _verify_pinned_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise C3NetConfigError(f"Pinned C3Net {label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise C3NetConfigError(
            f"Pinned C3Net {label} SHA256 mismatch: expected {expected_sha256}, got {observed}."
        )


def _runtime_identity_payload() -> dict[str, object]:
    """Return the exact executable package/build identity used by C3Net."""

    import numpy as np
    import torch
    from rdkit import rdBase

    try:
        numpy_configuration = np.show_config(mode="dicts")
    except TypeError as exc:
        raise C3NetConfigError(
            "C3Net formal runtime requires auditable NumPy build metadata."
        ) from exc
    numpy_configuration_sha256 = hashlib.sha256(
        json.dumps(
            numpy_configuration,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    try:
        torch_build_settings = next(
            line.strip()
            for line in torch.__config__.show().splitlines()
            if "Build settings:" in line
        )
    except StopIteration as exc:
        raise C3NetConfigError(
            "C3Net formal runtime lacks auditable Torch build settings."
        ) from exc
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "torch": torch.__version__,
        },
        "platform": {
            "byteorder": sys.byteorder,
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "build_fingerprints": {
            "numpy_config_sha256": numpy_configuration_sha256,
            "rdkit_build": rdBase.rdkitBuild,
            "torch_build_settings_sha256": hashlib.sha256(
                torch_build_settings.encode("utf-8")
            ).hexdigest(),
        },
    }


class C3NetPropertyAdapter:
    """Execute the exact C3Net property predictor in an isolated subprocess.

    ``source_root`` must be an unmodified checkout of the official C3Net
    source at :data:`C3NET_SOURCE_REVISION`.  MAPLE neither copies nor changes
    the GPL-licensed upstream source or its pretrained artifacts.
    """

    def __init__(
        self,
        source_root: str | Path,
        *,
        timeout_seconds: float = 120.0,
        runtime_manifest_path: str | Path | None = None,
    ):
        root = Path(source_root).expanduser().resolve()
        if not root.is_dir():
            raise C3NetConfigError(f"C3Net source_root is not a directory: {root}")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise C3NetConfigError("timeout_seconds must be finite and positive.")
        self.source_root = root
        self.timeout_seconds = float(timeout_seconds)
        self.runtime_manifest_path = (
            Path(runtime_manifest_path).expanduser().resolve()
            if runtime_manifest_path is not None
            else None
        )
        self.assert_source_identity()
        self.assert_runtime_identity()

    def assert_runtime_identity(self) -> None:
        """Fail closed when a formal runtime differs from its observed lock."""

        if self.runtime_manifest_path is None:
            return
        if not self.runtime_manifest_path.is_file():
            raise C3NetConfigError(
                "C3Net runtime manifest is missing: " f"{self.runtime_manifest_path}"
            )
        try:
            manifest = json.loads(
                self.runtime_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise C3NetConfigError("C3Net runtime manifest is invalid.") from exc
        if not isinstance(manifest, Mapping):
            raise C3NetConfigError("C3Net runtime manifest must be a JSON object.")
        expected_controls = {
            "deterministic_algorithms": True,
            "interop_threads": 1,
            "matmul_precision": "highest",
            "numpy_random_seed": 0,
            "python_random_seed": 0,
            "torch_random_seed": 0,
            "torch_threads": 1,
        }
        if manifest.get("required_controls") != expected_controls:
            raise C3NetConfigError(
                "C3Net runtime manifest lacks the exact deterministic controls."
            )
        expected_identity = {
            field: manifest.get(field)
            for field in ("python", "packages", "platform", "build_fingerprints")
        }
        observed_identity = _runtime_identity_payload()
        if expected_identity != observed_identity:
            raise C3NetConfigError(
                "C3Net executable runtime differs from the registered observed "
                "runtime lock."
            )

    def assert_source_identity(self) -> None:
        """Reject any source checkout other than the pinned official release."""

        if (
            _git_output(self.source_root, "rev-parse", "--is-inside-work-tree")
            != "true"
        ):
            raise C3NetConfigError("C3Net source_root must be a git work tree.")
        revision = _git_output(self.source_root, "rev-parse", "HEAD")
        if revision != C3NET_SOURCE_REVISION:
            raise C3NetConfigError(
                "C3Net source revision mismatch: "
                f"expected {C3NET_SOURCE_REVISION}, got {revision}."
            )
        origin = _normalize_origin(
            _git_output(self.source_root, "config", "--get", "remote.origin.url")
        )
        if origin != _normalize_origin(C3NET_SOURCE_URL):
            raise C3NetConfigError(
                f"C3Net source must have official origin {C3NET_SOURCE_URL}, got {origin!r}."
            )
        if _git_output(
            self.source_root, "status", "--porcelain", "--untracked-files=all"
        ):
            raise C3NetConfigError(
                "C3Net source checkout has tracked or untracked modifications; "
                "restore the pinned source."
            )
        for diff_args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            result = _git_result(self.source_root, *diff_args)
            if result.returncode == 1:
                raise C3NetConfigError(
                    "C3Net source checkout has tracked modifications; restore the pinned source."
                )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise C3NetConfigError(
                    f"Unable to verify the C3Net source checkout ({' '.join(diff_args)}): {detail}"
                )
        _verify_pinned_file(
            self.source_root / "prediction/model/checkpoint-1.pth.tar",
            C3NET_CHECKPOINT_SHA256,
            "prediction checkpoint",
        )
        _verify_pinned_file(
            self.source_root / "module/input/embedding.pt",
            C3NET_EMBEDDING_SHA256,
            "embedding artifact",
        )

    def available_solvents(self) -> tuple[str, ...]:
        """Return the exact 103 upstream solvent identifiers (not aliases)."""

        self.assert_source_identity()
        self.assert_runtime_identity()
        payload = self._run_script(_SOLVENTS_SCRIPT)
        raw_solvents = payload.get("solvents")
        if not isinstance(raw_solvents, list):
            raise C3NetRuntimeError(
                "C3Net solvent registry returned an invalid payload."
            )
        solvents = tuple(str(item).strip() for item in raw_solvents)
        if (
            len(solvents) != C3NET_SOLVENT_COUNT
            or len(set(solvents)) != C3NET_SOLVENT_COUNT
            or not all(solvents)
            or "water" not in solvents
            or {"logp", "pampa"}.intersection(solvents)
        ):
            raise C3NetRuntimeError(
                "Pinned C3Net source did not expose the expected 103-solvent registry."
            )
        return solvents

    def predict(self, sdf_path: str | Path, solvent: str) -> C3NetPropertyResult:
        """Predict one neutral, single-fragment SDF record in one named solvent."""

        return self._predict(sdf_path, solvent, conformer_mode="single")

    def predict_conformers(
        self, sdf_path: str | Path, solvent: str
    ) -> C3NetPropertyResult:
        """Average one to five upstream C3Net conformer predictions.

        The arithmetic mean follows the C3Net paper's reported multiple-
        conformer evaluation convention.  It remains a scalar property result,
        not a configurational free-energy estimator or an ASE calculation.
        """

        return self._predict(sdf_path, solvent, conformer_mode="up_to_five")

    def _predict(
        self, sdf_path: str | Path, solvent: str, *, conformer_mode: str
    ) -> C3NetPropertyResult:

        path = Path(sdf_path).expanduser().resolve()
        if not path.is_file():
            raise C3NetConfigError(f"C3Net SDF input is not a file: {path}")
        solvent_key = str(solvent).strip().lower()
        if not solvent_key:
            raise C3NetConfigError(
                "C3Net solvent must be an explicit upstream identifier."
            )
        if solvent_key in {"logp", "pampa"}:
            raise C3NetConfigError(
                "C3Net logp and pampa heads are not solvation-free-energy solvents."
            )
        self.assert_source_identity()
        self.assert_runtime_identity()
        payload = self._run_script(
            _PREDICT_SCRIPT, str(path), solvent_key, conformer_mode
        )
        try:
            returned_solvent = str(payload["solvent"])
            solvent_id = int(payload["solvent_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise C3NetRuntimeError(
                "C3Net upstream predictor returned an invalid result payload."
            ) from exc
        raw_predictions = payload.get("predictions_kcal_mol")
        if raw_predictions is None and "prediction_kcal_mol" in payload:
            raw_predictions = [payload["prediction_kcal_mol"]]
        if not isinstance(raw_predictions, list):
            raise C3NetRuntimeError(
                "C3Net upstream predictor returned invalid conformer predictions."
            )
        try:
            predictions = tuple(float(value) for value in raw_predictions)
        except (TypeError, ValueError) as exc:
            raise C3NetRuntimeError(
                "C3Net upstream predictor returned invalid conformer predictions."
            ) from exc
        expected_count = 1 if conformer_mode == "single" else None
        if (
            not predictions
            or len(predictions) > 5
            or (expected_count is not None and len(predictions) != expected_count)
            or not all(math.isfinite(value) for value in predictions)
        ):
            raise C3NetRuntimeError(
                "C3Net upstream predictor returned an invalid conformer prediction set."
            )
        if returned_solvent != solvent_key or not 0 <= solvent_id < C3NET_SOLVENT_COUNT:
            raise C3NetRuntimeError(
                "C3Net upstream predictor returned a mismatched solvent identity."
            )
        shim = payload.get("numpy_int_compatibility_shim", False)
        if not isinstance(shim, bool):
            raise C3NetRuntimeError(
                "C3Net upstream predictor returned an invalid NumPy compatibility flag."
            )
        return C3NetPropertyResult(
            predicted_solvation_free_energy_kcal_mol=math.fsum(predictions)
            / len(predictions),
            solvent=returned_solvent,
            solvent_id=solvent_id,
            conformer_policy=(
                "one supplied SDF record; upstream n_conformer=1"
                if conformer_mode == "single"
                else "one to five supplied SDF records; arithmetic mean of upstream predictions"
            ),
            conformer_count=len(predictions),
            component_predictions_kcal_mol=predictions,
            numpy_int_compatibility_shim=shim,
        )

    def _run_script(self, script: str, *arguments: str) -> Mapping[str, object]:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        inherited_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(self.source_root)
        if inherited_path:
            environment["PYTHONPATH"] += os.pathsep + inherited_path
        try:
            result = subprocess.run(
                [sys.executable, "-c", script, *arguments],
                cwd=self.source_root / "prediction",
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise C3NetRuntimeError(
                f"C3Net upstream predictor exceeded its {self.timeout_seconds:g}-second timeout."
            ) from exc
        except OSError as exc:
            raise C3NetRuntimeError(
                "Unable to start the C3Net upstream predictor."
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise C3NetRuntimeError(f"C3Net upstream predictor failed: {detail}")
        for line in reversed(result.stdout.splitlines()):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                return payload
        raise C3NetRuntimeError(
            "C3Net upstream predictor did not emit a JSON result payload."
        )

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from ase import Atoms

from maple.function.dispatcher.solvfe.protocol import (
    RouteAProtocol,
    canonical_sha256,
)

from .supermolecule_pcm import (
    SupermoleculePCMRule,
    sphere_union_report_dict,
)


class Route2SourceMismatch(RuntimeError):
    """The selected Route 2 checkout does not match the frozen contract."""


class Route2OuterRuntimeError(RuntimeError):
    """The frozen Route 2 worker failed or rejected at least one frame."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FrozenRoute2SMDBackend:
    """Audited process boundary to the frozen Route 2 SMD implementation.

    Route A owns neither the MACE-POLAR/PCM coupling nor the SMD CDS formula.
    It serializes same-topology frames, verifies an exact clean Route 2 source
    snapshot, and runs that implementation in a child interpreter.  One child
    handles a whole batch so the official MACE-POLAR model is loaded once.
    """

    def __init__(
        self,
        *,
        contract_path: str | Path,
        source_root: str | Path | None = None,
        audit_root: str | Path,
        pcmsolver_library: str | Path,
        pcmsolver_python_path: str | Path,
        model_checkpoint: str | Path,
        device: str = "cuda",
        timeout_seconds: float = 3600.0,
        python_executable: str | Path = sys.executable,
    ) -> None:
        self.contract_path = Path(contract_path).resolve()
        self.contract = json.loads(
            self.contract_path.read_text(encoding="utf-8")
        )
        parity = self.contract["route2_parity_source"]
        self.source_root = Path(
            source_root if source_root is not None else parity["repository"]
        ).resolve()
        self.audit_root = Path(audit_root).resolve()
        self.pcmsolver_library = Path(pcmsolver_library).resolve()
        self.pcmsolver_python_path = Path(pcmsolver_python_path).resolve()
        self.model_checkpoint = Path(model_checkpoint).resolve()
        self.device = str(device)
        self.timeout_seconds = float(timeout_seconds)
        self.python_executable = Path(python_executable).resolve()
        self.worker_path = Path(__file__).with_name(
            "_route2_outer_worker.py"
        ).resolve()
        self._verify_contract_and_runtime()

    def _git(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(self.source_root), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    def _verify_contract_and_runtime(self) -> None:
        if self.contract.get("contract_name") != (
            "route-a-polar-smd-outer-adapter"
        ):
            raise Route2SourceMismatch("Unexpected Route A outer contract.")
        if not self.source_root.is_dir():
            raise Route2SourceMismatch(
                f"Frozen Route 2 source is not a directory: {self.source_root}"
            )
        parity = self.contract["route2_parity_source"]
        try:
            observed_commit = self._git("rev-parse", "HEAD")
            dirty = self._git("status", "--porcelain")
        except subprocess.CalledProcessError as exc:
            raise Route2SourceMismatch(
                f"Unable to inspect frozen Route 2 source: {exc.stderr.strip()}"
            ) from exc
        if observed_commit != parity["commit"]:
            raise Route2SourceMismatch(
                "Frozen Route 2 commit mismatch: "
                f"expected {parity['commit']}, observed {observed_commit}."
            )
        if parity.get("clean_tree_required") is True and dirty:
            raise Route2SourceMismatch(
                "Frozen Route 2 source must have a clean working tree."
            )
        for relative_path, expected_hash in parity["source_sha256"].items():
            path = self.source_root / relative_path
            if not path.is_file() or _file_sha256(path) != expected_hash:
                raise Route2SourceMismatch(
                    f"Frozen Route 2 source hash mismatch: {relative_path}."
                )

        expected_pcm_hash = self.contract["pcm_response"][
            "pcmsolver_library_sha256"
        ]
        if (
            not self.pcmsolver_library.is_file()
            or _file_sha256(self.pcmsolver_library) != expected_pcm_hash
        ):
            raise Route2SourceMismatch(
                "PCMSolver library does not match the frozen outer contract."
            )
        if not (self.pcmsolver_python_path / "pcmsolver").is_dir():
            raise Route2SourceMismatch(
                "PCMSolver Python parser path does not contain pcmsolver/."
            )
        if not self.model_checkpoint.is_file():
            raise Route2SourceMismatch(
                f"MACE-POLAR checkpoint is missing: {self.model_checkpoint}"
            )
        if not self.python_executable.is_file():
            raise Route2SourceMismatch(
                f"Python executable is missing: {self.python_executable}"
            )
        if not self.worker_path.is_file():
            raise Route2SourceMismatch(
                f"Route 2 worker is missing: {self.worker_path}"
            )

    @staticmethod
    def _topology(atoms: Atoms) -> dict[str, Any]:
        mol2 = atoms.info.get("mol2")
        atom_types = (
            mol2.get("atom_types") if isinstance(mol2, dict) else None
        )
        if atom_types is None or len(atom_types) != len(atoms):
            raise ValueError(
                "Frozen GAFF2-O outer evaluation requires one MOL2 atom type "
                "per atom."
            )
        return {
            "atomic_numbers": [int(value) for value in atoms.numbers],
            "atom_types": [str(value) for value in atom_types],
            "charge": int(atoms.info.get("charge", 0)),
            "multiplicity": int(atoms.info.get("mult", 1)),
            "pbc": [bool(value) for value in atoms.pbc],
        }

    def _request(self, atoms: list[Atoms]) -> tuple[dict[str, Any], np.ndarray]:
        if not atoms:
            raise ValueError("Route 2 outer batch cannot be empty.")
        topology = self._topology(atoms[0])
        positions: list[np.ndarray] = []
        frame_ids: list[str] = []
        for index, frame in enumerate(atoms):
            if self._topology(frame) != topology:
                raise ValueError(
                    "STATE_SPACE_MISMATCH: outer batch topology changed at "
                    f"frame {index}."
                )
            coordinates = np.asarray(frame.positions, dtype=np.float64)
            if coordinates.shape != (len(frame), 3) or not np.all(
                np.isfinite(coordinates)
            ):
                raise ValueError(
                    f"Outer frame {index} coordinates are invalid."
                )
            positions.append(coordinates.copy())
            frame_ids.append(
                canonical_sha256(
                    {
                        "atomic_numbers": topology["atomic_numbers"],
                        "positions_angstrom": coordinates.tolist(),
                    }
                )
            )

        parity = self.contract["route2_parity_source"]
        source_hash = canonical_sha256(parity["source_sha256"])
        request = {
            "schema_version": 1,
            "contract_path": str(self.contract_path),
            "contract_sha256": canonical_sha256(self.contract),
            "source_root": str(self.source_root),
            "source_commit": parity["commit"],
            "source_sha256": parity["source_sha256"],
            "source_blob_sha256": source_hash,
            "provider_config": self.contract["provider_config"],
            "cavity_profile": self.contract["cavity_profile"],
            "topology": topology,
            "frame_ids": frame_ids,
            "runtime": {
                "device": self.device,
                "adapter_source_sha256": _file_sha256(Path(__file__)),
                "worker_source_sha256": _file_sha256(self.worker_path),
                "model_checkpoint": str(self.model_checkpoint),
                "model_sha256": _file_sha256(self.model_checkpoint),
                "pcmsolver_library": str(self.pcmsolver_library),
                "pcmsolver_library_sha256": _file_sha256(
                    self.pcmsolver_library
                ),
                "pcmsolver_python_path": str(self.pcmsolver_python_path),
                "python_executable": str(self.python_executable),
            },
        }
        request["request_sha256"] = canonical_sha256(request)
        return request, np.stack(positions)

    def __call__(self, atoms: Atoms) -> dict[str, Any]:
        return self.evaluate_many([atoms])[0]

    def evaluate_many(self, atoms: list[Atoms]) -> list[dict[str, Any]]:
        request, positions = self._request(atoms)
        run_dir = self.audit_root / request["request_sha256"]
        request_path = run_dir / "request.json"
        positions_path = run_dir / "positions.npz"
        output_path = run_dir / "output.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        request = {
            **request,
            "positions_path": str(positions_path),
            "output_path": str(output_path),
            "frame_audit_root": str(run_dir / "frames"),
        }
        _atomic_json(request_path, request)
        _atomic_npz(positions_path, positions_angstrom=positions)

        if output_path.is_file():
            output = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                output.get("request_sha256") == request["request_sha256"]
                and output.get("status") == "complete"
            ):
                return self._validated_worker_results(request, output)

        environment = os.environ.copy()
        conda_prefix = environment.get("CONDA_PREFIX")
        if conda_prefix:
            conda_lib = str(Path(conda_prefix) / "lib")
            existing = environment.get("LD_LIBRARY_PATH")
            environment["LD_LIBRARY_PATH"] = (
                conda_lib if not existing else f"{conda_lib}:{existing}"
            )
        environment["PCMSOLVER_LIBRARY"] = str(self.pcmsolver_library)
        environment["PCMSOLVER_PYTHON_PATH"] = str(
            self.pcmsolver_python_path
        )
        environment["PYTHONPATH"] = str(self.source_root)

        completed = subprocess.run(
            [
                str(self.python_executable),
                str(self.worker_path),
                str(request_path),
            ],
            cwd=self.source_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
        )
        (run_dir / "worker.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (run_dir / "worker.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not output_path.is_file():
            raise Route2OuterRuntimeError(
                "Frozen Route 2 outer worker failed; see "
                f"{run_dir.relative_to(self.audit_root)}."
            )
        output = json.loads(output_path.read_text(encoding="utf-8"))
        return self._validated_worker_results(request, output)

    @staticmethod
    def _validated_worker_results(
        request: dict[str, Any],
        output: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if output.get("request_sha256") != request["request_sha256"]:
            raise Route2OuterRuntimeError(
                "Frozen Route 2 worker returned a different request hash."
            )
        results = output.get("results")
        if not isinstance(results, list) or len(results) != len(
            request["frame_ids"]
        ):
            raise Route2OuterRuntimeError(
                "Frozen Route 2 worker returned the wrong frame count."
            )
        successful: list[dict[str, Any]] = []
        for index, (frame_id, result) in enumerate(
            zip(request["frame_ids"], results, strict=True)
        ):
            if result.get("frame_id") != frame_id:
                raise Route2OuterRuntimeError(
                    f"Frozen Route 2 frame identity mismatch at index {index}."
                )
            if result.get("status") != "ok":
                error = result.get("error", {})
                raise Route2OuterRuntimeError(
                    "Frozen Route 2 rejected frame "
                    f"{index}: {error.get('type', 'Error')}: "
                    f"{error.get('message', 'unknown failure')}."
                )
            successful.append(result["value"])
        return successful


class FrozenRoute2SupermoleculePCMBackend(FrozenRoute2SMDBackend):
    """Route A v2 electrostatic outer backend on the frozen Route 2 stack.

    The official MACE-POLAR/PCM coupling and source provenance remain frozen
    at the Route 2 boundary.  Route A changes only the preregistered
    whole-supermolecule cavity construction and excludes CDS from the returned
    Hamiltonian.
    """

    def __init__(
        self,
        *,
        protocol_path: str | Path,
        contract_path: str | Path,
        source_root: str | Path | None = None,
        audit_root: str | Path,
        pcmsolver_library: str | Path,
        pcmsolver_python_path: str | Path,
        model_checkpoint: str | Path,
        device: str = "cuda",
        timeout_seconds: float = 3600.0,
        python_executable: str | Path = sys.executable,
    ) -> None:
        self.protocol = RouteAProtocol.load(protocol_path)
        self.cavity_rule = SupermoleculePCMRule.from_protocol(self.protocol)
        if self.cavity_rule.cds_policy != (
            "excluded-from-v2-core-mandatory-ablation"
        ):
            raise ValueError(
                "Route A v2 supermolecule PCM requires CDS exclusion."
            )
        super().__init__(
            contract_path=contract_path,
            source_root=source_root,
            audit_root=audit_root,
            pcmsolver_library=pcmsolver_library,
            pcmsolver_python_path=pcmsolver_python_path,
            model_checkpoint=model_checkpoint,
            device=device,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )

    def _request(
        self,
        atoms: list[Atoms],
    ) -> tuple[dict[str, Any], np.ndarray]:
        request, positions = super()._request(atoms)
        request.pop("request_sha256", None)

        reference_fragments = self.cavity_rule.fragment_ids(atoms[0])
        reports: list[dict[str, Any]] = []
        for index, frame in enumerate(atoms):
            fragment_ids = self.cavity_rule.fragment_ids(frame)
            if not np.array_equal(fragment_ids, reference_fragments):
                raise ValueError(
                    "STATE_SPACE_MISMATCH: inferred fragment identities "
                    f"changed at frame {index}; provide a frozen "
                    "atoms.arrays['solvfe_fragment_id'] mapping."
                )
            reports.append(
                sphere_union_report_dict(self.cavity_rule.preflight(frame))
            )

        request.update(
            {
                "provider_mode": (
                    "route-a-scaled-supermolecule-pcm-electrostatic-v2"
                ),
                "route_a_protocol": {
                    "path": str(self.protocol.path),
                    "sha256": self.protocol.content_hash,
                    "version": self.protocol.data["protocol_version"],
                },
                "supermolecule_cavity": self.cavity_rule.as_request(),
                "scaled_radii_angstrom": (
                    self.cavity_rule.radii_angstrom(atoms[0]).tolist()
                ),
                "fragment_ids": reference_fragments.tolist(),
                "frame_cavity_preflight": reports,
            }
        )
        request["request_sha256"] = canonical_sha256(request)
        return request, positions

    @staticmethod
    def _validated_worker_results(
        request: dict[str, Any],
        output: dict[str, Any],
    ) -> list[dict[str, Any]]:
        values = FrozenRoute2SMDBackend._validated_worker_results(
            request,
            output,
        )
        for index, value in enumerate(values):
            warnings = value.get("warnings")
            if not isinstance(warnings, list) or warnings:
                raise Route2OuterRuntimeError(
                    "Route A v2 supermolecule PCM rejected frame "
                    f"{index}: native and PEDRA warnings are fatal and frame "
                    "deletion is forbidden."
                )
        return values

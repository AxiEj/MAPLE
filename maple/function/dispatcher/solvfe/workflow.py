from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ase import Atoms
from ase.data import covalent_radii

from .artifacts import ArtifactStore
from .protocol import RouteAProtocol, canonical_sha256, raw_sha256
from .states import CalculatorSpec, SolvFERequest


def _atom_list_hash(atoms: Atoms) -> str:
    return canonical_sha256(
        {
            "atom_count": len(atoms),
            "atomic_numbers": [int(value) for value in atoms.numbers],
        }
    )


def _coordinate_hash(atoms: Atoms) -> str:
    return canonical_sha256(
        {
            "atomic_numbers": [int(value) for value in atoms.numbers],
            "positions_angstrom": atoms.get_positions().tolist(),
            "cell_angstrom": atoms.cell.array.tolist(),
            "pbc": [bool(value) for value in atoms.get_pbc()],
        }
    )


def _is_connected(atoms: Atoms) -> bool:
    if len(atoms) <= 1:
        return True
    positions = atoms.get_positions()
    adjacency = [set() for _ in atoms]
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            cutoff = 1.25 * (
                covalent_radii[int(atoms.numbers[left])]
                + covalent_radii[int(atoms.numbers[right])]
            )
            distance = float(
                ((positions[left] - positions[right]) ** 2).sum() ** 0.5
            )
            if distance <= cutoff:
                adjacency[left].add(right)
                adjacency[right].add(left)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == len(atoms)


def _verify_checkpoint(spec: CalculatorSpec) -> None:
    path = Path(spec.checkpoint).expanduser()
    if not path.is_file():
        raise ValueError(
            f"MODEL_HASH_MISMATCH: checkpoint for role '{spec.role}' "
            f"does not exist: {path}"
        )
    actual = raw_sha256(path)
    if actual != spec.sha256:
        raise ValueError(
            f"MODEL_HASH_MISMATCH: checkpoint for role '{spec.role}' "
            f"declares {spec.sha256} but hashes to {actual}."
        )


def _calculator_spec(
    *,
    role: str,
    name: str,
    options: Mapping[str, Any],
    protocol_model: Mapping[str, Any],
    capabilities: frozenset[str],
    verify_checkpoint: bool,
) -> CalculatorSpec:
    expected_hash = str(protocol_model["checkpoint_sha256"])
    declared_hash = str(options.get("sha256", expected_hash))
    if declared_hash != expected_hash:
        raise ValueError(
            f"MODEL_HASH_MISMATCH: role '{role}' must use the protocol-locked "
            f"SHA {expected_hash}."
        )
    checkpoint = str(options.get("checkpoint", protocol_model["checkpoint"]))
    spec = CalculatorSpec(
        role=role,
        name=name,
        checkpoint=checkpoint,
        sha256=declared_hash,
        license_acknowledged=options.get("license_ack") is True,
        capabilities=capabilities,
        options=options,
    )
    if verify_checkpoint:
        _verify_checkpoint(spec)
    return spec


@dataclass(frozen=True)
class PreparedSolvFE:
    request: SolvFERequest
    protocol: RouteAProtocol
    sampler: CalculatorSpec
    scorer: CalculatorSpec
    outer: CalculatorSpec
    atom_list_hash: str
    coordinate_hash: str
    run_hash: str
    manifest: Mapping[str, Any]


class SolvationFreeEnergyWorkflow:
    """Dependency-ordered Route A workflow.

    Phase 1 implements the fail-closed PRECHECK/dry-run boundary.  Production
    stages are intentionally unavailable until their provider and estimator
    contracts are implemented; a dry run never loads model weights.
    """

    def __init__(
        self,
        *,
        prepared: PreparedSolvFE,
        atoms: Atoms,
        primary_calculator: Any,
        output: str | Path,
    ) -> None:
        self.prepared = prepared
        self.atoms = atoms.copy()
        self.primary_calculator = primary_calculator
        self.output = Path(output)

    @classmethod
    def prepare(
        cls,
        *,
        params: Mapping[str, Any],
        atoms: Atoms,
        project_root: str | Path,
    ) -> PreparedSolvFE:
        if not isinstance(atoms, Atoms):
            raise ValueError(
                "DOMAIN_UNSUPPORTED: Route A requires exactly one input solute."
            )
        if any(bool(value) for value in atoms.get_pbc()):
            raise ValueError(
                "DOMAIN_UNSUPPORTED: Route A input solute must be non-periodic."
            )
        if atoms.info.get("charge") != 0 or atoms.info.get("mult") != 1:
            raise ValueError(
                "DOMAIN_UNSUPPORTED: Route A v1 requires explicit charge=0 "
                "and mult=1 metadata."
            )
        if not _is_connected(atoms):
            raise ValueError(
                "DOMAIN_UNSUPPORTED: Route A v1 requires one connected solute."
            )

        request = SolvFERequest.from_params(params)
        root = Path(project_root).resolve()
        protocol_path = Path(request.protocol)
        if not protocol_path.is_absolute():
            protocol_path = root / protocol_path
        protocol = RouteAProtocol.load(protocol_path, project_root=root)
        domain = protocol.data["domain"]
        if (
            domain["solvent"]["name"] != request.solvent
            or domain["temperature_k"] != request.temperature
            or domain["pressure_bar"] != request.pressure_bar
        ):
            raise ValueError(
                "DOMAIN_UNSUPPORTED: request does not match the frozen Route A protocol."
            )
        allowed = set(domain["solute_constraints"]["allowed_elements"])
        unsupported = sorted(set(atoms.get_chemical_symbols()) - allowed)
        if unsupported:
            raise ValueError(
                "DOMAIN_UNSUPPORTED: unsupported element(s): "
                + ", ".join(unsupported)
            )

        model_options = params.get("model_options")
        scorer_options = params.get("scorer_options")
        outer_options = params.get("outer_options")
        if not all(
            isinstance(value, Mapping)
            for value in (model_options, scorer_options, outer_options)
        ):
            raise ValueError(
                "MODEL_CAPABILITY_MISSING: sampler, scorer, and outer specs "
                "must all be present."
            )

        verify_models = not request.dry_run
        protocol_models = protocol.data["models"]
        sampler = _calculator_spec(
            role="sampler",
            name=str(params["model"]),
            options=model_options,
            protocol_model=protocol_models["sampler"],
            capabilities=frozenset({"energy", "forces", "pbc"}),
            verify_checkpoint=verify_models,
        )
        scorer = _calculator_spec(
            role="target",
            name=str(params["scorer"]),
            options=scorer_options,
            protocol_model=protocol_models["target"],
            capabilities=frozenset(
                {"energy", "forces", "charge", "multiplicity"}
            ),
            verify_checkpoint=verify_models,
        )
        outer = _calculator_spec(
            role="outer",
            name=str(params["outer"]),
            options=outer_options,
            protocol_model=protocol_models["outer"],
            capabilities=frozenset(
                {"energy_delta", "whole_cluster", "standard_state_1m"}
            ),
            verify_checkpoint=False,
        )

        atom_hash = _atom_list_hash(atoms)
        coordinate_hash = _coordinate_hash(atoms)
        run_preimage = {
            "request": request.as_dict(),
            "protocol_hash": protocol.content_hash,
            "atom_list_hash": atom_hash,
            "coordinate_hash": coordinate_hash,
            "model_hashes": {
                "sampler": sampler.sha256,
                "target": scorer.sha256,
                "outer": outer.sha256,
            },
        }
        run_hash = canonical_sha256(run_preimage)
        manifest = {
            "schema_version": 1,
            "artifact_type": "route-a-run-manifest",
            "run_hash": run_hash,
            "request": request.as_dict(),
            "protocol": {
                "path": protocol.path.as_posix(),
                "sha256": protocol.content_hash,
                "version": protocol.data["protocol_version"],
            },
            "solute": {
                "atom_list_hash": atom_hash,
                "coordinate_hash": coordinate_hash,
                "charge": 0,
                "multiplicity": 1,
                "pbc": False,
            },
            "models": {
                "sampler": sampler.as_dict(),
                "target": scorer.as_dict(),
                "outer": outer.as_dict(),
            },
        }
        return PreparedSolvFE(
            request=request,
            protocol=protocol,
            sampler=sampler,
            scorer=scorer,
            outer=outer,
            atom_list_hash=atom_hash,
            coordinate_hash=coordinate_hash,
            run_hash=run_hash,
            manifest=manifest,
        )

    @property
    def run_directory(self) -> Path:
        return self.output.with_suffix(".solvfe")

    def run(self) -> dict[str, Any]:
        prepared = self.prepared
        store = ArtifactStore(
            self.run_directory,
            run_hash=prepared.run_hash,
        )
        store.initialize(dict(prepared.manifest))
        store.write_json("protocol.json", prepared.protocol.data)
        store.write_json(
            "models.json",
            {
                "sampler": prepared.sampler.as_dict(),
                "target": prepared.scorer.as_dict(),
                "outer": prepared.outer.as_dict(),
            },
        )
        store.write_json(
            "environment.json",
            {
                "schema_version": 1,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pymbar_available": importlib.util.find_spec("pymbar")
                is not None,
            },
        )

        precheck = store.resume_stage(
            "PRECHECK",
            input_hash=prepared.run_hash,
        )
        if precheck is None:
            precheck = {
                "schema_version": 1,
                "stage": "PRECHECK",
                "status": "completed",
                "run_hash": prepared.run_hash,
                "protocol_sha256": prepared.protocol.content_hash,
                "verified_artifact_count": (
                    prepared.protocol.verified_artifact_count
                ),
                "dry_run": prepared.request.dry_run,
            }
            store.complete_stage(
                "PRECHECK",
                input_hash=prepared.run_hash,
                output=precheck,
            )

        if not prepared.request.dry_run:
            if importlib.util.find_spec("pymbar") is None:
                raise RuntimeError(
                    "PYMBAR_MISSING: install PyMBAR 4 before production Route A execution."
                )
            raise NotImplementedError(
                "Route A production sampling is fail-closed until the staged "
                "QCT/MBAR runtime is implemented."
            )

        result = {
            "schema_version": 1,
            "method": "qct",
            "status": "engineering-ready",
            "claim": "research-qct-hybrid",
            "delta_g_hyd_kcal_mol": None,
            "standard_error_kcal_mol": None,
            "temperature_k": prepared.request.temperature,
            "standard_state": "1M(gas)->1M(solution)",
            "hamiltonian_id": prepared.protocol.data["hamiltonian_id"],
            "shell_boundaries_angstrom": [],
            "occupancies": [],
            "diagnostics": {
                "precheck_only": True,
                "verified_artifact_count": (
                    prepared.protocol.verified_artifact_count
                ),
            },
            "provenance": {
                "run_hash": prepared.run_hash,
                "protocol_sha256": prepared.protocol.content_hash,
                "model_spec_hashes": {
                    "sampler": prepared.sampler.content_hash,
                    "target": prepared.scorer.content_hash,
                    "outer": prepared.outer.content_hash,
                },
            },
            "failure_reasons": [],
        }
        result_input_hash = canonical_sha256(precheck)
        existing = store.resume_stage("RESULT", input_hash=result_input_hash)
        if existing is None:
            store.complete_stage(
                "RESULT",
                input_hash=result_input_hash,
                output=result,
            )
        else:
            result = existing
        store.write_json("result.json", result)
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(
                "Route A #solvfe dry-run PRECHECK completed: "
                f"status={result['status']}, run_hash={prepared.run_hash}\n"
            )
            handle.write(
                f"Authoritative result: {self.run_directory / 'result.json'}\n"
            )
        return result

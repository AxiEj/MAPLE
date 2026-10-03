#!/usr/bin/env python3
"""Analytic Gaussian-CHA H/HVP qualification and bounded workflow runner."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import time
from typing import Any

import numpy as np
from ase import Atoms

ROOT = Path("/home/axie/MAPLE/MAPLE-implicitsolv-route1")
WORKTREE = ROOT / ".omx/worktrees/cha-freq-ts-v1-20261002"
OUTPUT_ROOT = ROOT / ".omx/benchmarks/route1-cha-freq-ts-v1-20261002"
SCRIPT = Path(__file__).resolve()
PROTOCOL_PATH = SCRIPT.with_name("cha_gaussian_freq_ts_protocol.json")
PROTOCOL_SHA256 = "e1f28ffdcc97f6909fb30a22e7aac243bf8dc6c4c1c653629d049ef0678997e1"
PLAN = ROOT / ".omx/plans/route1-cha-freq-ts-v1-20261002.md"
PLAN_SHA256 = "2c872a6c3035b307098b7631434dfc8c88b6ace6bb91ca86d8361d1f15ade07c"
HANDOFF = ROOT / ".omx/plans/route1-cha-freq-ts-v1-handoff.json"
HANDOFF_SHA256 = "5278dd634e7f24e21e39c814ffc1a7cfbb911aabcae4395a9b0eee4fca8f392b"
PROFILE_ID = (
    "gaussian-cha-r6-derivative-v2-numerical-profile-" "20261001.2-direct-complement"
)
KCAL_PER_HARTREE = 627.5094740631


class GraphTrace:
    """Runner-owned attempted graph/call accounting, including failures."""

    def __init__(self) -> None:
        self._operations: dict[str, dict[str, int | str]] = {}
        self._observed: dict[str, dict[str, int | str]] = {}
        self._operation_id = "unattributed"
        self._codes: dict[Any, str] = {}
        self.partial_samples: list[dict[str, Any]] = []

    @contextmanager
    def sample(self, kind: str, identity: dict[str, Any]):
        entry = {"kind": kind, **json_safe(identity), "status": "STARTED"}
        self.partial_samples.append(entry)
        try:
            yield entry
        except Exception as exc:
            entry.update(status="FAILED", error=_safe_error(exc))
            raise
        else:
            entry["status"] = "RETURNED"

    @staticmethod
    def _empty_operation(operation_id: str) -> dict[str, int | str]:
        return {
            "operation_id": operation_id,
            "gas_graphs": 0,
            "solvent_graphs": 0,
            "independent_energy_calls": 0,
        }

    def _observe(self, frame, event, arg):
        if event == "call":
            key = self._codes.get(frame.f_code)
            if key is not None:
                record = self._observed.setdefault(
                    self._operation_id, self._empty_operation(self._operation_id)
                )
                record[key] = int(record[key]) + 1

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError("graph trace requires an unoccupied profile hook")
        self._codes = _graph_code_map()
        sys.setprofile(self._observe)
        return self

    def __exit__(self, exc_type, exc, traceback):
        sys.setprofile(None)

    def attempt(
        self,
        operation_id: str,
        *,
        gas_graphs: int = 0,
        solvent_graphs: int = 0,
        independent_energy_calls: int = 0,
    ) -> None:
        self._operation_id = operation_id
        self._observed.setdefault(operation_id, self._empty_operation(operation_id))
        record = self._operations.setdefault(
            operation_id,
            {
                "operation_id": operation_id,
                "gas_graphs": 0,
                "solvent_graphs": 0,
                "independent_energy_calls": 0,
            },
        )
        record["gas_graphs"] = int(record["gas_graphs"]) + int(gas_graphs)
        record["solvent_graphs"] = int(record["solvent_graphs"]) + int(solvent_graphs)
        record["independent_energy_calls"] = int(
            record["independent_energy_calls"]
        ) + int(independent_energy_calls)

    def receipt(self) -> dict[str, Any]:
        operations = [dict(self._observed[key]) for key in sorted(self._observed)]
        declared = [dict(self._operations[key]) for key in sorted(self._operations)]
        return {
            key: sum(int(operation[key]) for operation in operations)
            for key in ("gas_graphs", "solvent_graphs", "independent_energy_calls")
        } | {
            "operations": operations,
            "declared_operations": declared,
            "annotations_match_observations": operations == declared,
            "trace_contract": (
                "observed Python call events at primitive graph/geometry seams; "
                "failed calls count, annotations are checked rather than trusted"
            ),
            "graph_seams": _graph_seams(),
        }


def _graph_code_map() -> dict[Any, str]:
    from maple.function.calculator.mace._mace_calculator import build_data_from_atoms
    from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_gaussian import (
        continuum_gaussian_cha_scalar,
    )

    oracle = importlib.import_module(
        "docs.implicit-solvation.benchmarks.cha_gaussian_reference"
    )
    return {
        build_data_from_atoms.__code__: "gas_graphs",
        continuum_gaussian_cha_scalar.__code__: "solvent_graphs",
        oracle.prepare_gaussian_reference_geometry.__code__: "independent_energy_calls",
    }


@lru_cache(maxsize=1)
def _graph_seams() -> dict[str, Any]:
    from maple.function.calculator.mace._mace_calculator import build_data_from_atoms
    from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_gaussian import (
        continuum_gaussian_cha_scalar,
    )

    functions = {
        "gas": build_data_from_atoms,
        "solvent": continuum_gaussian_cha_scalar,
    }
    result = {}
    for name, function in functions.items():
        source_name = inspect.getsourcefile(function)
        if source_name is None:
            raise RuntimeError(f"{name} graph seam has no source")
        source = Path(source_name).resolve()
        result[name] = {
            "module": function.__module__,
            "qualname": function.__qualname__,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "dtype_contract": "CPU torch.float64",
        }
    oracle = importlib.import_module(
        "docs.implicit-solvation.benchmarks.cha_gaussian_reference"
    )
    function = oracle.prepare_gaussian_reference_geometry
    result["independent"] = {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "source": str(Path(inspect.getfile(function)).resolve()),
        "source_sha256": sha256_file(Path(inspect.getfile(function))),
        "dtype_contract": "NumPy float64/SciPy epsabs=epsrel=1e-12",
    }
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def content_sha256(value: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
    )


def seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = content_sha256(result)
    return result


def verify_seal(value: dict[str, Any], label: str) -> None:
    if value.get("content_sha256") != content_sha256(value):
        raise ValueError(f"{label} content seal differs")


def read_hashed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, digest


def read_externally_pinned_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256:
        raise ValueError(f"{label} external SHA256 pin differs")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, digest


def write_json_new_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def coordinate_sha256(positions: Any) -> str:
    values = np.asarray(positions, dtype="<f8")
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError("water coordinates must have finite shape (3,3)")
    return sha256_bytes(values.tobytes(order="C"))


def _base_runner():
    return importlib.import_module(
        "docs.implicit-solvation.benchmarks.run_cha_gaussian_opt"
    )


def load_validation_helper():
    benchmark = str(SCRIPT.parent)
    if benchmark not in sys.path:
        sys.path.insert(0, benchmark)
    return importlib.import_module("cha_gaussian_second_derivative_validation")


def load_protocol() -> tuple[dict[str, Any], str]:
    protocol, digest = read_hashed_json(PROTOCOL_PATH, "FREQ/TS protocol")
    if digest != PROTOCOL_SHA256:
        raise ValueError("FREQ/TS protocol SHA256 changed")
    if protocol.get("approved_plan_sha256") != PLAN_SHA256:
        raise ValueError("protocol plan pin differs")
    if protocol.get("approved_handoff_sha256") != HANDOFF_SHA256:
        raise ValueError("protocol handoff pin differs")
    if protocol.get("profile_id") != PROFILE_ID:
        raise ValueError("protocol numerical profile differs")
    if len(protocol.get("cases", ())) != 15:
        raise ValueError("protocol must enumerate exactly 15 cells")
    return protocol, digest


def output_allowed(output: Path) -> bool:
    resolved = output.resolve()
    return bool(
        resolved.parent == OUTPUT_ROOT.resolve()
        and re.fullmatch(r"campaign-\d{8}T\d{6}Z", resolved.name)
    )


def validate_workflow_options(
    workflow: str, values: dict[str, Any], protocol: dict[str, Any]
) -> None:
    key = {
        "frequency": "frequency_options",
        "prfo": "prfo_options",
        "dimer": "dimer_options",
    }.get(workflow)
    if key is None:
        raise ValueError(f"unknown workflow: {workflow}")
    allowed = set(protocol["closed_option_keys"][workflow])
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {workflow} options: {', '.join(unknown)}")
    expected = dict(protocol[key])
    if workflow == "dimer" and "n_given" in values:
        expected["n_given"] = values["n_given"]
    for name, expected_value in expected.items():
        if name not in values or values[name] != expected_value:
            raise ValueError(f"{workflow} option {name} differs from protocol")


def assert_effective_workflow_params(
    workflow: str, params: Any, requested: dict[str, Any]
) -> None:
    for name, expected in requested.items():
        if name == "method" and workflow != "frequency":
            continue
        observed = getattr(params, name, None)
        if isinstance(expected, list):
            if not np.array_equal(np.asarray(observed), np.asarray(expected)):
                raise ValueError(f"effective {workflow} parameter {name} differs")
        elif observed != expected:
            raise ValueError(f"effective {workflow} parameter {name} differs")


def validate_dimer_execution(
    *,
    before: dict[str, int],
    after: dict[str, int],
    maximum_hvp_calls: int,
    derivative_mode: str,
    replay_calls: int = 0,
) -> None:
    keys = {
        "direct_hvp_calls",
        "gas_dense_hessian_calls",
        "solvent_dense_hessian_calls",
    }
    if any(
        set(value) != keys or any(type(x) is not int or x < 0 for x in value.values())
        for value in (before, after)
    ):
        raise ValueError("Dimer counter schema differs")
    hvp_calls = after["direct_hvp_calls"] - before["direct_hvp_calls"]
    if derivative_mode != "hvp":
        raise ValueError("Dimer did not use direct analytic HVP mode")
    if (
        hvp_calls < 1
        or replay_calls < 0
        or hvp_calls + replay_calls > maximum_hvp_calls
    ):
        raise ValueError("Dimer HVP call budget differs")
    if any(after[key] != before[key] for key in keys - {"direct_hvp_calls"}):
        raise ValueError("Dimer direct HVP materialized a dense Hessian")


def _source_snapshot() -> dict[str, str]:
    base = _base_runner()
    # Reuse the dependency roster, never another worktree's byte hashes/context.
    paths = set((WORKTREE / "maple").rglob("*.py"))
    paths.update(WORKTREE / relative for relative in base.BENCHMARK_SOURCE_MODULES)
    extra = [
        SCRIPT,
        PROTOCOL_PATH,
        WORKTREE
        / "maple/function/calculator/extra_correction/implicit/gaussian_cha_analytic_correction.py",
        WORKTREE / "maple/function/calculator/mace/_mace_cha_analytic_calculator.py",
        WORKTREE
        / "docs/implicit-solvation/benchmarks/cha_gaussian_second_derivative_validation.py",
        WORKTREE
        / "docs/implicit-solvation/benchmarks/cha_gaussian_workflow_validation.py",
        WORKTREE / "maple/function/dispatcher/ts/ts.py",
    ]
    paths.update(extra)
    paths.update((WORKTREE / "tests" / "solvation").glob("test*gaussian*.py"))
    paths.update((WORKTREE / "tests").glob("test_prfo*.py"))
    snapshot = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        snapshot[str(path.relative_to(WORKTREE))] = sha256_file(path)
    return dict(sorted(snapshot.items()))


def _input_snapshot(protocol: dict[str, Any]) -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL_PATH,
        "plan": PLAN,
        "handoff": HANDOFF,
        "frozen_opt_manifest": ROOT / protocol["frozen_opt_manifest"],
        "topology": ROOT / protocol["topology_path"],
        "checkpoint": WORKTREE / "maple/function/calculator/model/maceoff23m.pt",
        "source_campaign_preregistration": ROOT
        / protocol["source_campaign"]
        / "preregistration.json",
        "source_campaign_validation": ROOT
        / protocol["source_campaign"]
        / "validation.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _assert_inputs(protocol: dict[str, Any], inputs: dict[str, str]) -> None:
    expected = {
        "protocol": PROTOCOL_SHA256,
        "plan": PLAN_SHA256,
        "handoff": HANDOFF_SHA256,
        "frozen_opt_manifest": protocol["frozen_opt_manifest_sha256"],
        "topology": protocol["topology_file_sha256"],
        "checkpoint": protocol["checkpoint_sha256"],
        "source_campaign_preregistration": protocol[
            "source_campaign_preregistration_sha256"
        ],
        "source_campaign_validation": protocol["source_campaign_validation_sha256"],
    }
    if inputs != expected:
        raise ValueError("FREQ/TS frozen input identity differs")
    manifest, _ = read_hashed_json(
        ROOT / protocol["frozen_opt_manifest"], "frozen 914-member manifest"
    )
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 914:
        raise ValueError("frozen OPT manifest must enumerate 914 members")
    for relative, expected_sha in files.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"frozen OPT evidence changed: {relative}")
    for case in protocol["cases"]:
        source = ROOT / case["source_opt_row"]
        if sha256_file(source) != case["source_opt_row_sha256"]:
            raise ValueError(f"source OPT row changed: {case['row_id']}")
        row, _ = read_hashed_json(source, "source OPT row")
        if (
            coordinate_sha256(row["retained_positions_angstrom"])
            != case["terminal_coordinate_sha256"]
        ):
            raise ValueError(f"source terminal geometry changed: {case['row_id']}")


def _identity(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _dimer_call_sites():
    """Bind call sites to the frozen sources without a protocol/runner hash cycle."""
    from maple.function.dispatcher.ts.algorithm.dimer import Dimer

    def sites(function, receiver, method):
        lines, first = inspect.getsourcelines(function)
        tree = ast.parse(textwrap.dedent("".join(lines)))
        source = Path(inspect.getfile(function))
        return [
            {
                "function": function.__name__,
                "line": first + node.lineno - 1,
                "source_sha256": sha256_file(source),
            }
            for node in sorted(ast.walk(tree), key=lambda n: getattr(n, "lineno", 0))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == receiver
        ]

    run_sites = sites(Dimer.run, "self", "_evaluate_derivatives")
    rotation_sites = sites(
        Dimer._rotate_minimize_kappa, "self", "_evaluate_derivatives"
    )
    replay_sites = sites(run_dimer_workflow, "replay_calculator", "get_hvp")
    if len(run_sites) != 3 or len(rotation_sites) != 1 or len(replay_sites) != 1:
        raise ValueError("Dimer source call-site inventory changed")
    return {
        "dimer_call_sites": dict(zip(("initial", "iteration", "final"), run_sites))
        | {"rotation": rotation_sites[0]},
        "dimer_replay_call_site": replay_sites[0],
    }


def preregister(output: Path) -> dict[str, Any]:
    if not output_allowed(output):
        raise ValueError(
            "output must be a unique direct campaign-YYYYMMDDTHHMMSSZ child"
        )
    if Path.cwd().resolve() != WORKTREE.resolve() or not SCRIPT.is_relative_to(
        WORKTREE
    ):
        raise RuntimeError("FREQ/TS runner origin or cwd differs")
    protocol, protocol_sha = load_protocol()
    inputs = _input_snapshot(protocol)
    _assert_inputs(protocol, inputs)
    source = _source_snapshot()
    record = seal(
        {
            "schema_version": 1,
            "phase": "PREREGISTERED",
            "protocol_sha256": protocol_sha,
            "source_before": source,
            "source_identity_sha256": _identity(source),
            "input_before": inputs,
            "input_identity_sha256": _identity(inputs),
            "rows": protocol["cases"],
            "runtime": _base_runner()._runtime_record(),
            "guard_tag": os.environ.get("CHA_GAUSSIAN_GUARD_TAG"),
            **_dimer_call_sites(),
            "status": "FREQ_TS_PREREGISTERED",
        }
    )
    write_json_new_atomic(output / "preregistration.json", record)
    return record


def load_preregistration(
    output: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    prereg, digest = read_externally_pinned_json(
        output / "preregistration.json", expected_sha256, "FREQ/TS preregistration"
    )
    verify_seal(prereg, "FREQ/TS preregistration")
    protocol, protocol_sha = load_protocol()
    if prereg.get("protocol_sha256") != protocol_sha:
        raise ValueError("preregistration protocol pin differs")
    source = _source_snapshot()
    inputs = _input_snapshot(protocol)
    _assert_inputs(protocol, inputs)
    if source != prereg.get("source_before") or inputs != prereg.get("input_before"):
        raise ValueError("source or input drift after preregistration")
    if prereg.get("rows") != protocol["cases"]:
        raise ValueError("preregistered row roster differs")
    if any(prereg.get(key) != value for key, value in _dimer_call_sites().items()):
        raise ValueError("Dimer source call sites changed after preregistration")
    return prereg, digest


def load_topology(protocol: dict[str, Any]):
    from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
        ContinuumChaTopology,
    )

    mapping, _ = read_hashed_json(ROOT / protocol["topology_path"], "topology")
    return mapping, ContinuumChaTopology.from_mapping(
        mapping, expected_content_sha256=protocol["topology_content_sha256"]
    )


def fresh_analytic_system(case: dict[str, Any], topology: Any):
    from maple.function.calculator.extra_correction.implicit.gaussian_cha_analytic_correction import (
        GaussianChaAnalyticCorrection,
    )
    from maple.function.calculator.mace._mace_cha_analytic_calculator import (
        MACEChaAnalyticCalculator,
    )

    atoms = Atoms(
        topology.atomic_numbers, positions=case["terminal_positions_angstrom"]
    )
    atoms.new_array(
        "_maple_implicit_atom_identity", np.asarray(topology.atom_ids, dtype=np.int64)
    )
    correction = GaussianChaAnalyticCorrection(
        atoms,
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=case["sigma_e"],
        order=64,
        numerical_profile_id=PROFILE_ID,
    )
    calculator = MACEChaAnalyticCalculator(
        correction=correction,
        model_path=WORKTREE / "maple/function/calculator/model/maceoff23m.pt",
        device="cpu",
        model="maceoff23m",
    )
    calculator.prepare_analytic_derivatives()
    if (
        getattr(calculator, "hessian", None) != "analytic"
        or getattr(calculator, "analytic_implicit_derivatives_admitted", False)
        is not True
        or correction.analytic_task_derivatives_admitted is not True
        or correction.mode != "fixed"
        or correction.inner_mode is not None
    ):
        raise RuntimeError("analytic Gaussian-CHA composition was not admitted")
    atoms.calc = calculator
    return atoms, correction, calculator


def _array(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float64, copy=False)
    except ImportError:
        pass
    return np.asarray(value, dtype=np.float64)


def _direction_records(
    directions: list[dict[str, Any]], hessian: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "direction_id": item["direction_id"],
            "direction": item["direction"],
            "hvp_hartree_per_angstrom2": (
                hessian @ np.asarray(item["direction"], dtype=np.float64)
            ).tolist(),
        }
        for item in directions
    ]


def _term_derivatives(
    case: dict[str, Any],
    topology: Any,
    directions: list[dict[str, Any]],
    trace: GraphTrace,
) -> dict[str, Any]:
    import torch

    from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_gaussian import (
        continuum_gaussian_cha_scalar,
    )

    trace.attempt("component-solvent-scalar", solvent_graphs=1)
    positions = torch.tensor(
        case["terminal_positions_angstrom"], dtype=torch.float64, requires_grad=True
    )
    scalar = continuum_gaussian_cha_scalar(
        positions,
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=case["sigma_e"],
        order=64,
        numerical_profile_id=PROFILE_ID,
    )
    terms = {
        "polar": scalar.polar_kcal_mol,
        "cavity": scalar.cavity_kcal_mol,
        "dispersion": scalar.dispersion_kcal_mol,
        "total": scalar.total_kcal_mol,
    }
    records = {}
    for name, energy in terms.items():
        gradient = torch.autograd.grad(
            energy, positions, create_graph=True, retain_graph=True
        )[0]
        direct_hvp = []
        for item in directions:
            vector = torch.as_tensor(
                np.asarray(item["direction"], dtype=np.float64).reshape(3, 3),
                dtype=torch.float64,
            )
            product = torch.autograd.grad(
                gradient,
                positions,
                grad_outputs=vector,
                retain_graph=True,
            )[0]
            direct_hvp.append(
                {
                    "direction_id": item["direction_id"],
                    "direction": item["direction"],
                    "hvp_kcal_mol_per_angstrom2": product.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .tolist(),
                }
            )
        rows = [
            torch.autograd.grad(
                component, positions, retain_graph=True, create_graph=False
            )[0].reshape(-1)
            for component in gradient.reshape(-1)
        ]
        hessian = torch.stack(rows).detach().cpu().numpy()
        records[name] = {
            "energy_kcal_mol": float(energy.detach()),
            "gradient_kcal_mol_per_angstrom": gradient.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist(),
            "hessian_kcal_mol_per_angstrom2": hessian.tolist(),
            "directional_hvp": direct_hvp,
        }
    return records


def _center_term(
    energy: float,
    forces: np.ndarray,
    hessian: np.ndarray,
    directions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "energy_hartree": float(energy),
        "gradient_hartree_per_angstrom": (-forces).reshape(-1).tolist(),
        "forces_hartree_per_angstrom": forces.reshape(3, 3).tolist(),
        "hessian_hartree_per_angstrom2": hessian.tolist(),
        "directional_hvp": _direction_records(directions, hessian),
    }


def _analytic_center(
    atoms: Atoms,
    correction: Any,
    calculator: Any,
    directions: list[dict[str, Any]],
    trace: GraphTrace,
    *,
    covariance: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace.attempt(
        (
            "covariance-2-transformed-composed-ef-h"
            if covariance
            else "center-composed-ef"
        ),
        gas_graphs=1,
        solvent_graphs=1,
    )
    composed_forces = _array(atoms.get_forces())
    composed_energy = float(atoms.get_potential_energy(force_consistent=True))
    if not covariance:
        trace.attempt("center-solvent-ef", solvent_graphs=1)
    else:
        trace.attempt("covariance-2-transformed-composed-ef-h", solvent_graphs=1)
    solvent = correction.evaluate(atoms, need_forces=True)
    solvent_forces = _array(solvent.forces_hartree_per_angstrom)
    trace.attempt(
        (
            "covariance-2-transformed-composed-ef-h"
            if covariance
            else "center-solvent-dense-h"
        ),
        solvent_graphs=1,
    )
    solvent_hessian = _array(correction.get_hessian(atoms))
    trace.attempt(
        (
            "covariance-2-transformed-composed-ef-h"
            if covariance
            else "center-composed-dense-h"
        ),
        gas_graphs=1,
        solvent_graphs=1,
    )
    composed_hessian = _array(calculator.get_hessian(atoms))
    gas_hessian = composed_hessian - solvent_hessian
    gas_energy = composed_energy - solvent.energy_hartree
    gas_forces = composed_forces - solvent_forces
    records = {
        "solvent": _center_term(
            solvent.energy_hartree,
            solvent_forces,
            solvent_hessian,
            directions,
        ),
        "gas": _center_term(gas_energy, gas_forces, gas_hessian, directions),
        "composed": _center_term(
            composed_energy, composed_forces, composed_hessian, directions
        ),
    }
    direct = []
    for item in directions:
        direction = np.asarray(item["direction"], dtype=np.float64)
        trace.attempt("center-18-composed-direct-hvp", gas_graphs=1, solvent_graphs=1)
        composed_hvp, tuple_forces, tuple_energy = calculator.get_hvp(atoms, direction)
        trace.attempt("center-18-separate-solvent-direct-hvp", solvent_graphs=1)
        solvent_directional = correction.get_directional_derivatives(atoms, direction)
        composed_hvp = _array(composed_hvp).reshape(-1)
        solvent_hvp = _array(solvent_directional.hvp_hartree_per_angstrom2).reshape(-1)
        gas_hvp = composed_hvp - solvent_hvp
        for name, values in (
            ("solvent", solvent_hvp),
            ("gas", gas_hvp),
            ("composed", composed_hvp),
        ):
            match = next(
                record
                for record in records[name]["directional_hvp"]
                if record["direction_id"] == item["direction_id"]
            )
            match["dense_hessian_product_hartree_per_angstrom2"] = match.pop(
                "hvp_hartree_per_angstrom2"
            )
            match["direct_hvp_hartree_per_angstrom2"] = values.tolist()
        direct.append(
            {
                "direction_id": item["direction_id"],
                "direction": item["direction"],
                "composed_hvp_hartree_per_angstrom2": composed_hvp.tolist(),
                "tuple_forces_hartree_per_angstrom": _array(tuple_forces)
                .reshape(-1)
                .tolist(),
                "tuple_energy_hartree": float(tuple_energy),
            }
        )
    identity = {}
    if direct:
        fresh = direct[0]
        identity = {
            "evaluation_count": 1,
            "fresh_gas_energy_hartree": gas_energy,
            "fresh_solvent_energy_hartree": solvent.energy_hartree,
            "fresh_composed_energy_hartree": composed_energy,
            "fresh_gas_forces_hartree_per_angstrom": gas_forces.tolist(),
            "fresh_solvent_forces_hartree_per_angstrom": solvent_forces.tolist(),
            "fresh_composed_forces_hartree_per_angstrom": composed_forces.tolist(),
            "hvp_tuple_energy_hartree": fresh["tuple_energy_hartree"],
            "hvp_tuple_forces_hartree_per_angstrom": fresh[
                "tuple_forces_hartree_per_angstrom"
            ],
        }
    return records, {"direct_hvp": direct, "fresh_identity": identity}


def _force_validation(
    case: dict[str, Any],
    atoms: Atoms,
    calculator: Any,
    protocol: dict[str, Any],
    trace: GraphTrace,
) -> list[dict[str, Any]]:
    base = np.asarray(case["terminal_positions_angstrom"], dtype=np.float64)
    result = []
    try:
        for flat in range(9):
            atom, axis = divmod(flat, 3)
            steps = []
            for step in protocol["force_fd_steps_angstrom"]:
                samples = []
                for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                    displaced = base.copy()
                    displaced[atom, axis] += sign * float(step)
                    atoms.set_positions(displaced)
                    calculator.reset()
                    trace.attempt(
                        "own-force-validation-54-composed-ef",
                        gas_graphs=1,
                        solvent_graphs=1,
                    )
                    with trace.sample(
                        "own-force",
                        {
                            "dof": flat,
                            "step_angstrom": step,
                            "sign": sign,
                            "positions_angstrom": displaced.tolist(),
                            "coordinate_sha256": coordinate_sha256(displaced),
                        },
                    ) as retained:
                        forces = _array(atoms.get_forces())
                        retained["forces_hartree_per_angstrom"] = forces.tolist()
                    samples.append((label, displaced, forces))
                plus = samples[0]
                minus = samples[1]
                steps.append(
                    {
                        "step_angstrom": float(step),
                        "plus_positions_angstrom": plus[1].tolist(),
                        "plus_coordinate_sha256": coordinate_sha256(plus[1]),
                        "plus_composed_forces_hartree_per_angstrom": plus[2].tolist(),
                        "minus_positions_angstrom": minus[1].tolist(),
                        "minus_coordinate_sha256": coordinate_sha256(minus[1]),
                        "minus_composed_forces_hartree_per_angstrom": minus[2].tolist(),
                    }
                )
            result.append(
                {"dof": flat, "direction": np.eye(9)[flat].tolist(), "steps": steps}
            )
    finally:
        atoms.set_positions(base)
        calculator.reset()
    return result


def _reference_evaluation(
    oracle: Any,
    positions: np.ndarray,
    topology_mapping: dict[str, Any],
    sigma_e: float,
    order: int,
    trace: GraphTrace,
    operation_id: str,
) -> tuple[dict[str, Any], Any]:
    trace.attempt(operation_id, independent_energy_calls=1)
    with trace.sample(
        "independent-energy",
        {
            "positions_angstrom": positions.tolist(),
            "coordinate_sha256": coordinate_sha256(positions),
            "order": order,
            "sigma_e": sigma_e,
            "epsabs": 1e-12,
            "epsrel": 1e-12,
        },
    ) as retained:
        geometry = oracle.prepare_gaussian_reference_geometry(
            positions,
            topology_mapping["effective_charges_e"],
            topology_mapping["cha_radii_angstrom"],
            topology_mapping["lj_rmin_angstrom"],
            topology_mapping["lj_epsilon_kcal_mol"],
            azimuth_orders=(order,),
            epsabs=1e-12,
            epsrel=1e-12,
        )
        reference = oracle.gaussian_cha_from_reference_geometry(
            geometry, sigma_e=sigma_e
        )
        metadata = _base_runner()._reference_geometry_metadata(geometry)
        metadata.pop("content_sha256")
        for name in (
            "positions_angstrom",
            "charges_e",
            "cha_radii_angstrom",
            "lj_rmin_angstrom",
            "lj_epsilon_kcal_mol",
            "inverse_cube_per_angstrom3",
            "inverse_born_per_angstrom",
            "effective_charges_e",
            "electrostatic_size_angstrom",
        ):
            metadata[name] = json_safe(getattr(geometry, name))
        metadata = seal(metadata)
        retained.update(
            total_energy_kcal_mol=float(reference.total_kcal_mol),
            component_energies_kcal_mol=dict(reference.energies_kcal_mol),
            reference_geometry_metadata=metadata,
        )
    errors = [
        float(level.inverse_cube_quad_error_estimate_per_angstrom3)
        for level in geometry.r6_levels
    ]
    branch = {
        "domain": "admitted",
        "r6_diagnostics": json_safe(geometry.r6_diagnostics),
        "nonpolar_diagnostics": json_safe(geometry.diagnostics),
        "geometry_identity_sha256": geometry.identity_sha256,
    }
    return (
        {
            "total_energy_kcal_mol": float(reference.total_kcal_mol),
            "component_energies_kcal_mol": dict(reference.energies_kcal_mol),
            "quadrature_error_estimates": errors,
            "branch_metadata": branch,
            "reference_geometry_metadata": metadata,
        },
        reference,
    )


def _reference_center_diagnostics(
    case: dict[str, Any], topology_mapping: dict[str, Any], trace: GraphTrace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oracle = importlib.import_module(
        "docs.implicit-solvation.benchmarks.cha_gaussian_reference"
    )
    positions = np.asarray(case["terminal_positions_angstrom"], dtype=np.float64)
    records = []
    for order in (64, 96, 128):
        raw, _ = _reference_evaluation(
            oracle,
            positions,
            topology_mapping,
            case["sigma_e"],
            order,
            trace,
            "independent-center-orders-64-96-128",
        )
        records.append(
            {
                "order": order,
                "epsabs": 1e-12,
                "epsrel": 1e-12,
                "coordinate_sha256": coordinate_sha256(positions),
                **raw,
            }
        )
    shared, _ = _reference_evaluation(
        oracle,
        positions,
        topology_mapping,
        case["sigma_e"],
        128,
        trace,
        "independent-shared-center-128",
    )
    return records, shared


def _independent_curvature(
    case: dict[str, Any],
    topology_mapping: dict[str, Any],
    shared_center: dict[str, Any],
    protocol: dict[str, Any],
    trace: GraphTrace,
) -> list[dict[str, Any]]:
    oracle = importlib.import_module(
        "docs.implicit-solvation.benchmarks.cha_gaussian_reference"
    )
    base = np.asarray(case["terminal_positions_angstrom"], dtype=np.float64)
    result = []
    for item in case["nonrigid_directions"]:
        direction = np.asarray(item["values"], dtype=np.float64).reshape(3, 3)
        steps = []
        for step in protocol["curvature_steps_angstrom"]:
            samples = []
            for multiplier in (-2, -1, 0, 1, 2):
                positions = base + multiplier * float(step) * direction
                if multiplier == 0:
                    raw = shared_center
                else:
                    raw, _ = _reference_evaluation(
                        oracle,
                        positions,
                        topology_mapping,
                        case["sigma_e"],
                        128,
                        trace,
                        "independent-directional-displacements",
                    )
                samples.append(
                    {
                        "multiplier": multiplier,
                        "positions_angstrom": positions.tolist(),
                        "coordinate_sha256": coordinate_sha256(positions),
                        "energy_kcal_mol": raw["total_energy_kcal_mol"],
                        "quadrature_error_estimate": max(
                            raw["quadrature_error_estimates"]
                        ),
                        "branch_metadata": raw["branch_metadata"],
                        "reference_geometry_metadata": raw[
                            "reference_geometry_metadata"
                        ],
                        "component_energies_kcal_mol": raw[
                            "component_energies_kcal_mol"
                        ],
                    }
                )
            steps.append(
                {
                    "step_angstrom": float(step),
                    "reference_order": 128,
                    "epsabs": 1e-12,
                    "epsrel": 1e-12,
                    "samples": samples,
                }
            )
        result.append(
            {
                "direction_id": item["direction_id"],
                "direction": item["values"],
                "steps": steps,
            }
        )
    return result


def _covariance_records(
    case: dict[str, Any],
    topology: Any,
    directions: list[dict[str, Any]],
    protocol: dict[str, Any],
    base_terms: dict[str, Any],
    trace: GraphTrace,
) -> list[dict[str, Any]]:
    base = np.asarray(case["terminal_positions_angstrom"], dtype=np.float64)
    records = []
    for transform in protocol["covariance_transforms"]:
        q = np.asarray(transform["Q"], dtype=np.float64)
        a = np.asarray(transform["a"], dtype=np.float64)
        transformed = base @ q.T + a
        transformed_case = {**case, "terminal_positions_angstrom": transformed.tolist()}
        atoms, correction, calculator = fresh_analytic_system(
            transformed_case, topology
        )
        transformed_terms, _ = _analytic_center(
            atoms, correction, calculator, [], trace, covariance=True
        )
        for name in ("solvent", "gas", "composed"):
            records.append(
                {
                    "term": name,
                    "transform_id": transform["transform_id"],
                    "Q": transform["Q"],
                    "a": transform["a"],
                    "transformed_positions_angstrom": transformed.tolist(),
                    "transformed_coordinate_sha256": coordinate_sha256(transformed),
                    "base": base_terms[name],
                    "transformed": transformed_terms[name],
                }
            )
    return records


def produce_derivative_receipt(
    case: dict[str, Any],
    topology_mapping: dict[str, Any],
    topology: Any,
    prereg: dict[str, Any],
    preregistration_sha256: str,
    protocol: dict[str, Any],
    trace: GraphTrace | None = None,
) -> dict[str, Any]:
    validation = load_validation_helper()
    directions = validation.build_direction_inventory(
        np.asarray(case["terminal_positions_angstrom"], dtype=np.float64),
        seed=int(case["direction_seed"]),
        nonrigid_directions=case["nonrigid_directions"],
    )
    trace = GraphTrace() if trace is None else trace
    with trace:
        atoms, correction, calculator = fresh_analytic_system(case, topology)
        solvent_terms = _term_derivatives(case, topology, directions, trace)
        analytic_center, identities = _analytic_center(
            atoms, correction, calculator, directions, trace
        )
        reference_center, shared_reference_center = _reference_center_diagnostics(
            case, topology_mapping, trace
        )
        force_validation = _force_validation(case, atoms, calculator, protocol, trace)
        independent_curvature = _independent_curvature(
            case, topology_mapping, shared_reference_center, protocol, trace
        )
        covariance = _covariance_records(
            case, topology, directions, protocol, analytic_center, trace
        )
        counters = {
            **trace.receipt(),
            "hvp_calls": int(getattr(calculator, "direct_hvp_call_count", 0)),
            "core_api_counters": {
                "calculator": json_safe(
                    getattr(calculator, "analytic_graph_counters", {})
                ),
                "correction": json_safe(
                    getattr(correction, "analytic_graph_counters", {})
                ),
            },
        }
        expected_operations = {
            item["operation_id"]: item for item in protocol["budgets"]["operations"]
        }
        observed_operations = {
            item["operation_id"]: item for item in counters["operations"]
        }
        if (
            observed_operations != expected_operations
            or not counters["annotations_match_observations"]
        ):
            raise RuntimeError("actual attempted graph-operation inventory differs")
        return seal(
            {
                "case_identity": case,
                "preregistration_sha256": preregistration_sha256,
                "protocol_sha256": PROTOCOL_SHA256,
                "source_identity_sha256": prereg["source_identity_sha256"],
                "input_identity_sha256": prereg["input_identity_sha256"],
                "profile_id": PROFILE_ID,
                "production_order": 64,
                "topology_sha256": topology.content_sha256,
                "checkpoint_sha256": protocol["checkpoint_sha256"],
                "analytic_identity_fingerprint": correction.analytic_identity_fingerprint,
                "analytic_identity_provenance": correction.analytic_identity_provenance,
                "units": {
                    "center_energy": "hartree",
                    "center_force": "hartree/angstrom",
                    "center_hessian": "hartree/angstrom^2",
                    "solvent_terms": "kcal/mol,angstrom",
                    "direction": "dimensionless",
                },
                "direction_inventory": directions,
                "solvent_terms": solvent_terms,
                "analytic_center": analytic_center,
                "force_validation": force_validation,
                "reference_center_diagnostics": reference_center,
                "independent_curvature": independent_curvature,
                "covariance": covariance,
                "fresh_identity": identities["fresh_identity"],
                "direct_hvp_records": identities["direct_hvp"],
                "counters": counters,
                "exceptions": [],
            }
        )


def _read_workflow_output(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _workflow_geometry(atoms, case, protocol):
    positions = atoms.get_positions().tolist()
    masses, numbers = atoms.get_masses().tolist(), atoms.get_atomic_numbers().tolist()
    if (
        positions != case["terminal_positions_angstrom"]
        or masses != protocol["masses_amu"]
        or numbers != protocol["atomic_numbers"]
    ):
        raise ValueError("workflow geometry/mass/atom identity differs")
    return {
        "initial_positions_angstrom": positions,
        "masses_amu": masses,
        "atomic_numbers": numbers,
        "true_ts_claim": False,
    }


def run_frequency_workflow(
    case: dict[str, Any], topology: Any, row_dir: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    from maple.function.dispatcher.frequency.frequency import Frequency, MWFrequency

    options = dict(protocol["frequency_options"])
    validate_workflow_options("frequency", options, protocol)
    atoms, _correction, calculator = fresh_analytic_system(case, topology)
    geometry = _workflow_geometry(atoms, case, protocol)
    fresh_forces = _array(atoms.get_forces())
    output = row_dir / "frequency.out"
    started = time.monotonic()
    driver = Frequency(str(output), atoms, paras={"freq": options})
    assert_effective_workflow_params("frequency", driver.params, options)
    driver.run()
    elapsed = time.monotonic() - started
    analyzer = MWFrequency(str(row_dir / "frequency-recompute.out"), atoms)
    hessian = _array(calculator.get_hessian(atoms))
    frequencies, modes = analyzer.compute_frequencies(hessian)
    return {
        "status": "EXECUTED",
        **geometry,
        "requested_options": options,
        "effective_options": {
            key: json_safe(getattr(driver.params, key)) for key in options
        },
        "elapsed_seconds": elapsed,
        "fresh_forces_hartree_per_angstrom": fresh_forces.tolist(),
        "hessian_hartree_per_angstrom2": hessian.tolist(),
        "frequencies_cm1": frequencies.tolist(),
        "modes_cart": modes.tolist(),
        "raw_output": _read_workflow_output(output),
    }


def run_prfo_workflow(
    case: dict[str, Any], topology: Any, row_dir: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
    from maple.function.dispatcher.frequency.frequency import MWFrequency

    options = dict(protocol["prfo_options"])
    validate_workflow_options("prfo", options, protocol)
    atoms, _correction, calculator = fresh_analytic_system(case, topology)
    geometry = _workflow_geometry(atoms, case, protocol)
    output = row_dir / "prfo.out"
    job = PRFO(str(output), atoms, paras={"prfo": options})
    assert_effective_workflow_params("prfo", job.params, options)
    initial_hessian = _array(calculator.get_hessian(atoms))
    analyzer = MWFrequency(str(row_dir / "prfo-modes.out"), atoms)
    initial_frequencies, _ = analyzer.compute_frequencies(initial_hessian)
    started = time.monotonic()
    returned = job.run()
    elapsed = time.monotonic() - started
    final_hessian = _array(calculator.get_hessian(returned))
    final_frequencies, _ = analyzer.compute_frequencies(final_hessian)
    text = _read_workflow_output(output)
    return {
        "status": "INTERFACE_EXECUTED_NOT_TS",
        **geometry,
        "requested_options": options,
        "effective_options": {
            name: getattr(job.params, name) for name in options if name != "method"
        },
        "elapsed_seconds": elapsed,
        "raw_return_positions_angstrom": returned.get_positions().tolist(),
        "raw_output": text,
        "raw_algorithm_converged": "Normal Termination" in text,
        "initial_composed_hessian_hartree_per_angstrom2": initial_hessian.tolist(),
        "final_composed_hessian_hartree_per_angstrom2": final_hessian.tolist(),
        "initial_frequencies_cm1": initial_frequencies.tolist(),
        "final_frequencies_cm1": final_frequencies.tolist(),
        "initial_material_negative_count": int(np.sum(initial_frequencies < -10.0)),
        "final_material_negative_count": int(np.sum(final_frequencies < -10.0)),
    }


def _counter_snapshot(calculator: Any) -> dict[str, int]:
    counters = calculator.analytic_graph_counters
    solvent = calculator.solvent_correction.analytic_graph_counters
    return {
        "direct_hvp_calls": counters["direct_hvp"],
        "gas_dense_hessian_calls": counters["gas_dense_hessian"],
        "solvent_dense_hessian_calls": solvent["dense_hessian"],
    }


def run_dimer_workflow(
    case: dict[str, Any], topology: Any, row_dir: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    from maple.function.dispatcher.ts.algorithm.dimer import Dimer
    from maple.function.calculator import calculator_base
    from maple.function.calculator.mace import _mace_calculator

    prohibited_calls = []

    options = {**protocol["dimer_options"], "n_given": case["dimer_n_given"]}
    validate_workflow_options("dimer", options, protocol)
    atoms, _correction, calculator = fresh_analytic_system(case, topology)
    geometry = _workflow_geometry(atoms, case, protocol)
    output = row_dir / "dimer.out"
    job = Dimer(str(output), atoms, paras={"dimer": options})
    assert_effective_workflow_params("dimer", job.params, options)
    before = _counter_snapshot(calculator)
    actual_initial_direction = np.asarray(job.n, dtype=np.float64).copy()
    started = time.monotonic()
    prohibited_codes = {
        function.__code__: function.__qualname__
        for function in (
            type(calculator).get_hessian,
            type(calculator)._analytic_hessian,
            type(_correction).get_hessian,
            calculator_base.numerical_hessian_from_atoms,
            _mace_calculator.hessian_via_double_autograd,
            Dimer._finite_difference_hn,
        )
    }

    directional_evaluations = []
    pending = {}
    hvp_code = type(calculator).get_hvp.__code__
    replay_calculator = None
    replay_call_site = {}
    source_identity = {
        "profile_id": PROFILE_ID,
        "checkpoint_sha256": calculator.checkpoint_sha256,
        "topology_sha256": topology.content_sha256,
        "gas_source_sha256": sha256_file(Path(inspect.getfile(type(calculator)))),
        "solvent_source_sha256": sha256_file(Path(inspect.getfile(type(_correction)))),
    }
    dimer_source_sha256 = sha256_file(Path(inspect.getfile(Dimer)))

    def observe_derivative_call(frame, event, arg):
        if event == "call" and frame.f_code in prohibited_codes:
            prohibited_calls.append(prohibited_codes[frame.f_code])
            raise RuntimeError("Direct-HVP Dimer attempted prohibited curvature")
        if (
            frame.f_code == hvp_code
            and event == "call"
            and frame.f_locals.get("self") is replay_calculator
        ):
            caller = frame.f_back
            replay_call_site.update(
                function=caller.f_code.co_name,
                line=caller.f_lineno,
                source_sha256=sha256_file(SCRIPT),
            )
        if frame.f_code == hvp_code and frame.f_locals.get("self") is calculator:
            if event == "call":
                caller = frame.f_back.f_back
                run_frame = caller if caller.f_code.co_name == "run" else caller.f_back
                if run_frame.f_locals.get("self") is not job:
                    raise RuntimeError("unexpected Dimer HVP caller")
                role = (
                    "rotation"
                    if caller.f_code.co_name == "_rotate_minimize_kappa"
                    else (
                        "final"
                        if "final_center" in run_frame.f_locals
                        else "iteration" if "it" in run_frame.f_locals else "initial"
                    )
                )
                coordinates = frame.f_locals["atoms"].get_positions()
                direction = _array(frame.f_locals["n"])
                entry = {
                    "evaluation_id": f"hvp-{len(directional_evaluations)}",
                    "ordinal": len(directional_evaluations),
                    "call_kind": role,
                    "iteration": int(run_frame.f_locals.get("it", 0)),
                    "positions_angstrom": coordinates.tolist(),
                    "coordinate_sha256": coordinate_sha256(coordinates),
                    "direction": direction.tolist(),
                    "direction_sha256": sha256_bytes(
                        np.asarray(direction, dtype="<f8").tobytes()
                    ),
                    "source_identity": source_identity,
                    "call_site": {
                        "function": caller.f_code.co_name,
                        "line": caller.f_lineno,
                        "source_sha256": dimer_source_sha256,
                    },
                    "status": "STARTED",
                }
                pending[id(frame)] = entry
                directional_evaluations.append(entry)
            elif event == "return":
                entry = pending.pop(id(frame))
                if arg is None:
                    entry["status"] = "FAILED"
                else:
                    hv, forces, energy = arg
                    entry.update(
                        status="RETURNED",
                        hessian_vector_hartree_per_angstrom2=_array(hv)
                        .reshape(-1)
                        .tolist(),
                        forces_hartree_per_angstrom=_array(forces)
                        .reshape(3, 3)
                        .tolist(),
                        energy_hartree=float(energy),
                    )

    if sys.getprofile() is not None:
        raise RuntimeError("Dimer derivative guard requires an unoccupied profile hook")
    # Observe calls without changing the prepared, identity-bound interfaces.
    sys.setprofile(observe_derivative_call)
    try:
        job.run()
        algorithm_after = _counter_snapshot(calculator)
        final = directional_evaluations[-1]
        if final["call_kind"] != "final" or final["status"] != "RETURNED":
            raise RuntimeError("missing final Dimer HVP evaluation")
        replay_case = {
            **case,
            "terminal_positions_angstrom": final["positions_angstrom"],
        }
        replay_atoms, _, replay_calculator = fresh_analytic_system(
            replay_case, topology
        )
        replay_hv, replay_force, replay_energy = replay_calculator.get_hvp(
            replay_atoms, np.asarray(final["direction"], dtype=np.float64)
        )
        fresh_force = _array(replay_atoms.get_forces())
        fresh_energy = float(replay_atoms.get_potential_energy(force_consistent=True))
        directional_replays = [
            {
                "evaluation_id": final["evaluation_id"],
                "comparison_kind": "same_implementation_replay",
                **{
                    key: final[key]
                    for key in (
                        "positions_angstrom",
                        "coordinate_sha256",
                        "direction",
                        "direction_sha256",
                        "source_identity",
                    )
                },
                "hessian_vector_hartree_per_angstrom2": _array(replay_hv)
                .reshape(-1)
                .tolist(),
                "forces_hartree_per_angstrom": _array(replay_force)
                .reshape(3, 3)
                .tolist(),
                "energy_hartree": float(replay_energy),
                "fresh_forces_hartree_per_angstrom": fresh_force.tolist(),
                "fresh_energy_hartree": fresh_energy,
                "counters": _counter_snapshot(replay_calculator),
                "call_site": dict(replay_call_site),
            }
        ]
    finally:
        sys.setprofile(None)
    if prohibited_calls:
        raise RuntimeError("Dimer caught a prohibited derivative attempt")
    elapsed = time.monotonic() - started
    after = algorithm_after
    validate_dimer_execution(
        before=before,
        after=after,
        maximum_hvp_calls=protocol["workflow_budgets"]["dimer_composed_hvp_calls"],
        derivative_mode=job.derivative_mode,
        replay_calls=len(directional_replays),
    )
    text = _read_workflow_output(output)
    printed_kappas = [
        float(value)
        for value in re.findall(r"Curvature \(kappa\):\s+([-+0-9.eE]+)", text)
    ]
    iterations = [
        entry for entry in directional_evaluations if entry["call_kind"] == "iteration"
    ]
    if len(printed_kappas) != len(iterations):
        raise RuntimeError("Dimer printed curvature inventory differs")
    return {
        "status": "INTERFACE_EXECUTED_NOT_TS",
        **geometry,
        "requested_options": options,
        "effective_options": {
            name: json_safe(getattr(job.params, name))
            for name in options
            if name != "method"
        },
        "actual_initial_direction": actual_initial_direction.tolist(),
        "elapsed_seconds": elapsed,
        "raw_return_positions_angstrom": job.atoms.get_positions().tolist(),
        "raw_output": text,
        "raw_algorithm_converged": "converged TS candidate" in text,
        "counter_before": before,
        "counter_after": after,
        "derivative_mode": job.derivative_mode,
        "prohibited_derivative_attempts": prohibited_calls,
        "directional_evaluations": directional_evaluations,
        "directional_replays": directional_replays,
        "directional_units": {
            "energy": "hartree",
            "forces": "hartree/angstrom",
            "hvp": "hartree/angstrom^2",
            "direction": "dimensionless",
        },
        "curvature_history_hartree_per_angstrom2": printed_kappas,
        "printed_curvature_entries": [
            {
                "iteration": entry["iteration"],
                "evaluation_id": entry["evaluation_id"],
                "directional_evaluation_index": entry["ordinal"],
                "printed_kappa": kappa,
            }
            for entry, kappa in zip(iterations, printed_kappas)
        ],
    }


def _safe_error(exc: Exception) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _check_phase_bindings(record, case, phase, prereg, preregistration_sha256):
    expected = {
        "case_identity": case,
        "phase": phase,
        "preregistration_sha256": preregistration_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": prereg["source_identity_sha256"],
        "input_identity_sha256": prereg["input_identity_sha256"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"phase receipt {field} binding differs")


def _parse_rss_status(text: str) -> dict[str, int]:
    fields = {}
    for line in text.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            parts = line.split()
            if len(parts) != 3 or parts[2] != "kB" or int(parts[1]) < 0:
                raise ValueError("RSS counters must be nonnegative Linux kB values")
            fields[parts[0]] = int(parts[1]) * 1024
    if set(fields) != {"VmRSS:", "VmHWM:"} or fields["VmHWM:"] < fields["VmRSS:"]:
        raise ValueError("RSS counters are missing or inconsistent")
    return {"rss_bytes": fields["VmRSS:"], "peak_bytes": fields["VmHWM:"]}


def _rss_status() -> dict[str, int]:
    return _parse_rss_status(Path("/proc/self/status").read_text(encoding="utf-8"))


def _reset_peak_rss() -> tuple[dict[str, int], dict[str, int]]:
    """Reset only this process's RSS high-water counter, never its memory.

    Linux documents clear_refs=5 in https://docs.kernel.org/filesystems/proc.html.
    Unsupported kernels/permissions fail rather than substitute sampled RSS or
    a lifetime high-water difference that could hide an operation's peak.
    """
    before = _rss_status()
    Path("/proc/self/clear_refs").write_text("5\n", encoding="ascii")
    reset = _rss_status()
    if reset["peak_bytes"] - reset["rss_bytes"] > 2 * 1024 * 1024:
        raise RuntimeError("RSS high-water reset could not be verified")
    return before, reset


def run_resource_operation(
    case: dict[str, Any],
    topology: Any,
    operation: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if operation not in {"energy-force", "dense-h", "direct-hvp"}:
        raise ValueError("unknown resource operation")
    import gc

    atoms, _correction, calculator = fresh_analytic_system(case, topology)
    gc.collect()
    before_reset, baseline = _reset_peak_rss()
    started = time.monotonic()
    if operation == "energy-force":
        forces = _array(atoms.get_forces()).reshape(-1)
        value = np.r_[float(atoms.get_potential_energy(force_consistent=True)), forces]
    elif operation == "dense-h":
        value = _array(calculator.get_hessian(atoms))
    else:
        direction = np.asarray(case["dimer_n_given"], dtype=np.float64)
        value = _array(calculator.get_hvp(atoms, direction)[0])
    elapsed = time.monotonic() - started
    after = _rss_status()
    incremental = max(0, after["peak_bytes"] - baseline["rss_bytes"])
    return {
        "status": "RESOURCE_MEASURED",
        "operation": operation,
        "measurement": "linux-proc-reset-high-water-v1",
        "pid": os.getpid(),
        "kernel_release": os.uname().release,
        "before_reset": before_reset,
        "reset_baseline": baseline,
        "after_operation": after,
        "baseline_rss_bytes": baseline["rss_bytes"],
        "peak_rss_bytes": after["peak_bytes"],
        "incremental_peak_bytes": incremental,
        "elapsed_seconds": elapsed,
        "result_shape": list(value.shape),
        "fresh_process_required_for_comparison": True,
        "maximum_incremental_rss_bytes": protocol["workflow_budgets"][
            "incremental_rss_bytes"
        ],
    }


def run_campaign(
    output: Path,
    expected_preregistration_sha256: str,
    *,
    phase: str = "all",
    row_id: str | None = None,
    operation: str | None = None,
) -> dict[str, Any]:
    if phase == "resource" and row_id is None:
        raise ValueError("resource phase requires a single explicit row ID")
    prereg, observed = load_preregistration(output, expected_preregistration_sha256)
    protocol, _ = load_protocol()
    topology_mapping, topology = load_topology(protocol)
    phases = ("derivatives", "freq", "prfo", "dimer")
    if phase != "all" and phase not in {*phases, "resource"}:
        raise ValueError(f"unknown phase: {phase}")
    selected = [case for case in protocol["cases"] if row_id in (None, case["row_id"])]
    if row_id is not None and len(selected) != 1:
        raise ValueError(f"unknown row ID: {row_id}")
    requested_phases = phases if phase == "all" else (phase,)
    if "resource" in requested_phases and operation not in {
        "energy-force",
        "dense-h",
        "direct-hvp",
    }:
        raise ValueError("resource phase requires a valid --operation")
    if phase != "resource" and operation is not None:
        raise ValueError("--operation is valid only for the resource phase")
    results = []
    for case in selected:
        row_dir = output / "work" / case["row_id"]
        row_dir.mkdir(parents=True, exist_ok=True)
        for selected_phase in requested_phases:
            receipt_name = (
                f"{case['row_id']}-{operation}.json"
                if selected_phase == "resource"
                else f"{case['row_id']}.json"
            )
            destination = output / selected_phase / receipt_name
            if destination.exists():
                record, _ = read_hashed_json(destination, "existing phase receipt")
                verify_seal(record, "existing phase receipt")
                _check_phase_bindings(record, case, selected_phase, prereg, observed)
                if (
                    selected_phase == "resource"
                    and record.get("operation") != operation
                ):
                    raise ValueError("resource receipt operation differs")
                results.append(record)
                continue
            trace = GraphTrace() if selected_phase == "derivatives" else None
            try:
                if selected_phase == "derivatives":
                    payload = produce_derivative_receipt(
                        case,
                        topology_mapping,
                        topology,
                        prereg,
                        observed,
                        protocol,
                        trace,
                    )
                elif selected_phase == "freq":
                    payload = run_frequency_workflow(case, topology, row_dir, protocol)
                elif selected_phase == "prfo":
                    payload = run_prfo_workflow(case, topology, row_dir, protocol)
                elif selected_phase == "dimer":
                    payload = run_dimer_workflow(case, topology, row_dir, protocol)
                else:
                    assert operation is not None
                    payload = run_resource_operation(
                        case, topology, operation, protocol
                    )
                record = seal(
                    {
                        **payload,
                        "phase": selected_phase,
                        "operation": operation,
                        "case_identity": case,
                        "preregistration_sha256": observed,
                        "protocol_sha256": PROTOCOL_SHA256,
                        "source_identity_sha256": prereg["source_identity_sha256"],
                        "input_identity_sha256": prereg["input_identity_sha256"],
                    }
                )
            except Exception as exc:
                record = seal(
                    {
                        "status": "FAILED",
                        "phase": selected_phase,
                        "operation": operation,
                        "case_identity": case,
                        "preregistration_sha256": observed,
                        "protocol_sha256": PROTOCOL_SHA256,
                        "source_identity_sha256": prereg["source_identity_sha256"],
                        "input_identity_sha256": prereg["input_identity_sha256"],
                        "error": _safe_error(exc),
                        "attempted_counters": (
                            None if trace is None else trace.receipt()
                        ),
                        "partial_samples": (
                            [] if trace is None else trace.partial_samples
                        ),
                    }
                )
            write_json_new_atomic(destination, record)
            results.append(record)
    summary = seal(
        {
            "status": (
                "PHASE_EXECUTED"
                if results and all(item.get("status") != "FAILED" for item in results)
                else "INTERFACE_INCOMPLETE"
            ),
            "phase": phase,
            "row_id": row_id,
            "operation": operation,
            "record_count": len(results),
            "records": [
                {
                    "row_id": item["case_identity"]["row_id"],
                    "phase": item["phase"],
                    "status": item.get("status"),
                }
                for item in results
            ],
        }
    )
    suffix = f"-{phase}-{row_id}" if row_id else f"-{phase}"
    if operation is not None:
        suffix += f"-{operation}"
    write_json_new_atomic(output / f"run-summary{suffix}.json", summary)
    return summary


def _resource_passes(record: dict[str, Any], protocol: dict[str, Any]) -> bool:
    """Recompute resource metrics from reset kernel counters, not a pass flag."""
    if record.get("status") != "RESOURCE_MEASURED":
        return False
    try:
        reset, after = record["reset_baseline"], record["after_operation"]
        values = [
            reset["rss_bytes"],
            reset["peak_bytes"],
            after["rss_bytes"],
            after["peak_bytes"],
        ]
        if any(type(value) is not int or value < 0 for value in values):
            return False
        increment = max(0, after["peak_bytes"] - reset["rss_bytes"])
        elapsed = float(record["elapsed_seconds"])
        return bool(
            record["measurement"] == "linux-proc-reset-high-water-v1"
            and type(record["pid"]) is int
            and record["pid"] > 0
            and reset["rss_bytes"] > 0
            and 0 <= reset["peak_bytes"] - reset["rss_bytes"] <= 2 * 1024 * 1024
            and after["peak_bytes"] >= max(reset["peak_bytes"], after["rss_bytes"])
            and record["baseline_rss_bytes"] == reset["rss_bytes"]
            and record["peak_rss_bytes"] == after["peak_bytes"]
            and record["incremental_peak_bytes"] == increment
            and increment <= protocol["workflow_budgets"]["incremental_rss_bytes"]
            and np.isfinite(elapsed)
            and 0 <= elapsed <= 300.0
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_campaign(
    output: Path, expected_preregistration_sha256: str, *, write_output: bool = True
) -> dict[str, Any]:
    prereg, observed = load_preregistration(output, expected_preregistration_sha256)
    protocol, _ = load_protocol()
    helper = load_validation_helper()
    validation_protocol = {
        **protocol,
        "source_identity_sha256": prereg["source_identity_sha256"],
        "input_identity_sha256": prereg["input_identity_sha256"],
        "dimer_call_sites": prereg["dimer_call_sites"],
        "dimer_replay_call_site": prereg["dimer_replay_call_site"],
    }
    workflow_validator = importlib.import_module("cha_gaussian_workflow_validation")
    validation_protocol["workflow_sources"] = {
        "gas_source_sha256": prereg["source_before"][
            "maple/function/calculator/mace/_mace_cha_analytic_calculator.py"
        ],
        "solvent_source_sha256": prereg["source_before"][
            "maple/function/calculator/extra_correction/implicit/gaussian_cha_analytic_correction.py"
        ],
        "dimer_source_sha256": prereg["source_before"][
            "maple/function/dispatcher/ts/algorithm/dimer.py"
        ],
    }
    derivative_records = []
    workflow_verdicts = {"freq": [], "prfo": [], "dimer": []}
    workflow_records: dict[str, list[dict[str, Any]]] = {
        "freq": [],
        "prfo": [],
        "dimer": [],
    }
    resource_records: list[tuple[dict[str, Any], ...]] = []
    for case in protocol["cases"]:
        record, _ = read_hashed_json(
            output / "derivatives" / f"{case['row_id']}.json", "derivative receipt"
        )
        verify_seal(record, "derivative receipt")
        _check_phase_bindings(record, case, "derivatives", prereg, observed)
        derivative_records.append(
            helper.validate_freq_ts_row(record, validation_protocol, case)
        )
        for phase in workflow_records:
            workflow, _ = read_hashed_json(
                output / phase / f"{case['row_id']}.json", f"{phase} receipt"
            )
            verify_seal(workflow, f"{phase} receipt")
            _check_phase_bindings(workflow, case, phase, prereg, observed)
            workflow_records[phase].append(workflow)
            workflow_verdicts[phase].append(
                workflow_validator.validate_workflow_receipt(
                    workflow,
                    phase,
                    case,
                    validation_protocol,
                    record.get("analytic_center", {}).get("composed", {}),
                )
            )
        dense, _ = read_hashed_json(
            output / "resource" / f"{case['row_id']}-dense-h.json",
            "dense-H resource receipt",
        )
        hvp, _ = read_hashed_json(
            output / "resource" / f"{case['row_id']}-direct-hvp.json",
            "direct-HVP resource receipt",
        )
        verify_seal(dense, "dense-H resource receipt")
        verify_seal(hvp, "direct-HVP resource receipt")
        energy_force, _ = read_hashed_json(
            output / "resource" / f"{case['row_id']}-energy-force.json",
            "E/F resource receipt",
        )
        verify_seal(energy_force, "E/F resource receipt")
        for item, operation in (
            (energy_force, "energy-force"),
            (dense, "dense-h"),
            (hvp, "direct-hvp"),
        ):
            _check_phase_bindings(item, case, "resource", prereg, observed)
            if item.get("operation") != operation:
                raise ValueError("resource receipt operation differs")
        resource_records.append((energy_force, dense, hvp))
    second = helper.summarize_freq_ts_rows(
        derivative_records, validation_protocol["cases"]
    )
    frequency_pass = sum(row["passed"] for row in workflow_verdicts["freq"])
    prfo_pass = sum(row["passed"] for row in workflow_verdicts["prfo"])
    dimer_pass = sum(row["passed"] for row in workflow_verdicts["dimer"])
    resource_pass = sum(
        all(_resource_passes(item, protocol) for item in (energy_force, dense, hvp))
        and len({item["pid"] for item in (energy_force, dense, hvp)}) == 3
        and hvp["incremental_peak_bytes"]
        <= dense["incremental_peak_bytes"] + 64 * 1024 * 1024
        for energy_force, dense, hvp in resource_records
    )
    source_after = _source_snapshot()
    inputs_after = _input_snapshot(protocol)
    stable = (
        source_after == prereg["source_before"]
        and inputs_after == prereg["input_before"]
    )
    passed = bool(
        second.get("status") == "SECOND_DERIVATIVE_VALIDATED"
        and frequency_pass == prfo_pass == dimer_pass == 15
        and resource_pass == 15
        and stable
    )
    result = seal(
        {
            "status": (
                "ANALYTIC_FREQ_VALIDATED_TS_INTERFACES_ONLY"
                if passed
                else (
                    "SECOND_DERIVATIVE_UNRESOLVED"
                    if second.get("status") != "SECOND_DERIVATIVE_VALIDATED"
                    else "INTERFACE_INCOMPLETE"
                )
            ),
            "preregistration_sha256": observed,
            "second_derivative": second,
            "workflow_validation": workflow_verdicts,
            "frequency_pass_count": frequency_pass,
            "prfo_interface_count": prfo_pass,
            "dimer_interface_count": dimer_pass,
            "resource_pass_count": resource_pass,
            "source_input_stable": stable,
            "true_ts_claim": False,
            "rows": derivative_records,
        }
    )
    if write_output:
        write_json_new_atomic(output / "validation.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preregister", "run", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name in ("run", "validate"):
            command.add_argument("--expected-preregistration-sha256", required=True)
        if name == "run":
            command.add_argument(
                "--phase",
                choices=("all", "derivatives", "freq", "prfo", "dimer", "resource"),
                default="all",
            )
            command.add_argument("--row-id")
            command.add_argument(
                "--operation", choices=("energy-force", "dense-h", "direct-hvp")
            )
    args = parser.parse_args(argv)
    if not output_allowed(args.output):
        parser.error("--output must be a unique direct campaign-YYYYMMDDTHHMMSSZ child")
    if args.command == "preregister":
        result = preregister(args.output)
        success = result.get("status") == "FREQ_TS_PREREGISTERED"
    elif args.command == "run":
        result = run_campaign(
            args.output,
            args.expected_preregistration_sha256,
            phase=args.phase,
            row_id=args.row_id,
            operation=args.operation,
        )
        success = result.get("status") == "PHASE_EXECUTED"
    else:
        result = validate_campaign(args.output, args.expected_preregistration_sha256)
        success = result.get("status") == "ANALYTIC_FREQ_VALIDATED_TS_INTERFACES_ONLY"
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Frozen, research-only upstream PCM comparison; never an admitted PES.

The MACE-POLAR checkpoint/source, SMD Coulomb radii, and legacy CDS stay fixed.
PySCF supplies the SWIG/ISWIG surface and Gaussian-charge C-PCM matrix.  The
candidate changes the electrostatic equation from ddPCM to C-PCM, so differences
from v3 are model changes, not implementation errors or experimental accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PANEL = (
    ROOT / ".omx/research/pure-torch-analytic-20260922/"
    "ten-molecule-panel-20260923/run-1"
)
CANARY = ROOT / "tools/route2_release/run_pure_mace_polar_torch_canary.py"
CHECKPOINT = Path.home() / ".cache/mace/MACE-POLAR-1-M.model"
ORDERS = {17: 110, 23: 194, 29: 302}
METHODS = ("SWIG", "ISWIG")
ROTATION_CASES = ("water", "methane", "hydrogen-cyanide")
FORCE_CASES = ("water", "methane", "hydrogen-cyanide")
AUDIT_CASES = ("water", "hydrogen-cyanide")
ROTATION_SEED = 20260923
ROTATION_COUNT = 4
AUDIT_STEPS_ANGSTROM = (1.0e-3, 1.0e-4)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _source_manifest() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("frozen_canary", CANARY)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen v3 source manifest builder.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._source_manifest()


def _source_manifest_digest() -> str:
    return hashlib.sha256(
        json.dumps(_source_manifest(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare() -> None:
    if (HERE / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen; do not overwrite it.")
    import pyscf

    if pyscf.__version__ != "2.13.1":
        raise RuntimeError("The benchmark is version-frozen to PySCF 2.13.1.")
    source = _source_manifest()
    reference = json.loads((PANEL / "source-manifest.json").read_text())
    if source != reference:
        raise RuntimeError(
            "Current MAPLE source differs from the ten-molecule v3 panel."
        )
    cases = json.loads((PANEL / "protocol.json").read_text())["cases"]
    if len(cases) != 10 or [case["name"] for case in cases] != [
        "water",
        "methane",
        "ammonia",
        "carbon-monoxide",
        "carbon-dioxide",
        "hydrogen-cyanide",
        "formaldehyde",
        "hydrogen-peroxide",
        "acetylene",
        "hydrogen-fluoride",
    ]:
        raise RuntimeError("The frozen ten-molecule case list changed.")
    protocol = {
        "status": "frozen-before-batch",
        "purpose": "upstream C-PCM numerical/model-change comparison, not physical accuracy",
        "case_names": [case["name"] for case in cases],
        "case_protocol_sha256": _sha(PANEL / "protocol.json"),
        "source_manifest_sha256": _source_manifest_digest(),
        "script_sha256": _sha(Path(__file__)),
        "checkpoint_sha256": _sha(CHECKPOINT),
        "pyscf_version": pyscf.__version__,
        "solvent": "water",
        "electrostatic_equation": "C-PCM; v3 reference is ddPCM",
        "cavity_radii": "unchanged SMD Coulomb radii by atom",
        "nonpolar": "unchanged Torch legacy SMD-CDS",
        "methods": METHODS,
        "lebedev_order_to_points": ORDERS,
        "rotation_cases": ROTATION_CASES,
        "rotation_seed": ROTATION_SEED,
        "rotation_count": ROTATION_COUNT,
        "force_cases": FORCE_CASES,
        "force_method": "SWIG only, existing analytic PySCF adapter",
        "force_orders": (17, 29),
        "audit_cases": AUDIT_CASES,
        "audit_steps_angstrom": AUDIT_STEPS_ANGSTROM,
        "metrics": (
            "candidate polar and total solvation energies; v3 polar difference;"
            " 110/194/302 grid stability; rigid-rotation energy spread;"
            " SWIG analytic force difference and directional FD audit"
        ),
        "fit_performed": False,
        "experimental_accuracy_tested": False,
        "selection_rule": "all ten cases retained including failures and absent v3 HCN reference",
    }
    _write(HERE / "protocol.json", protocol)


def _atoms(case: dict) -> Atoms:
    atoms = Atoms(case["symbols"], positions=case["positions_angstrom"])
    atoms.info.update(charge=case["charge"], mult=case["multiplicity"])
    return atoms


def _rotated_source(raw: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = raw.copy()
    dipoles_xyz = raw[:, 1:4][:, [2, 0, 1]]
    result[:, 1:4] = (dipoles_xyz @ rotation.T)[:, [1, 2, 0]]
    return result


def _cpcm(
    atoms: Atoms,
    raw: np.ndarray,
    radii: np.ndarray,
    dielectric: float,
    method: str,
    order: int,
) -> dict:
    from pyscf import gto
    from pyscf.solvent import pcm
    from maple.function.calculator.extra_correction.implicit.gto_density import (
        point_multipole_potential,
    )
    from maple.function.calculator.extra_correction.implicit.pyscf_swig_response import (
        _AtomIndexedSurfaceMolecule,
    )

    started = time.perf_counter()
    molecule = gto.M(
        atom=list(
            zip(atoms.get_chemical_symbols(), atoms.positions.tolist(), strict=True)
        ),
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )
    surface = pcm.gen_surface(
        _AtomIndexedSurfaceMolecule(molecule),
        ng=ORDERS[order],
        rad=radii / Bohr,
        surface_discretization_method=method,
    )
    _, matrix = pcm.get_D_S(surface, with_S=True, with_D=False)
    matrix = np.asarray(matrix, dtype=float)
    potential = point_multipole_potential(
        np.asarray(surface["grid_coords"]), atoms.positions, raw
    )
    factor = (dielectric - 1.0) / dielectric
    charge = lu_solve(lu_factor(matrix, check_finite=True), -factor * potential)
    residual = float(
        np.linalg.norm(matrix @ charge + factor * potential)
        / max(np.linalg.norm(factor * potential), 1.0e-14)
    )
    energy = 0.5 * float(np.dot(potential, charge)) * Hartree
    area = np.asarray(surface["area"], dtype=float)
    return {
        "polar_energy_eV": energy,
        "surface_points": int(len(potential)),
        "area_A2": float(area.sum() * Bohr**2),
        "relative_solve_residual": residual,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _swig_polar_gradient(
    atoms: Atoms, positions, learned, radii, dielectric, order: int
) -> tuple[float, np.ndarray]:
    import torch
    from maple.function.calculator.extra_correction.implicit.gto_density import (
        density_to_external_field_order,
        external_field_to_density_order,
        point_multipole_potential,
    )
    from maple.function.calculator.extra_correction.implicit.pyscf_swig_response import (
        PySCFSWIGCPCMResponse,
    )

    raw = learned.detach().cpu().numpy()
    response = PySCFSWIGCPCMResponse(
        tuple(atoms.get_chemical_symbols()),
        atoms.positions,
        radii,
        dielectric=dielectric,
        lebedev_order=order,
    )
    potential = point_multipole_potential(
        response.surface_points_bohr, atoms.positions, raw
    )
    energy = response.solve(potential).polarization_energy_hartree * Hartree
    reaction = response.reaction_field_linear_map()
    field = reaction.apply(raw)
    source_cotangent = torch.tensor(
        external_field_to_density_order(field),
        dtype=positions.dtype,
        device=positions.device,
    )
    source_gradient = (
        torch.autograd.grad(
            learned, positions, grad_outputs=source_cotangent, retain_graph=True
        )[0]
        .detach()
        .cpu()
        .numpy()
    )
    explicit_gradient = 0.5 * reaction.full_position_vjp(
        raw, density_to_external_field_order(raw)
    )
    return float(energy), source_gradient + explicit_gradient


def run() -> None:
    import pyscf
    import torch
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.models.mace_polar import build_official_mace_polar_1_m_adapter
    from maple.solvation.models.mace_polar_torch import MACEPolarTorchGraphAdapter
    from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

    protocol = json.loads((HERE / "protocol.json").read_text())
    if (
        protocol["script_sha256"] != _sha(Path(__file__))
        or protocol["checkpoint_sha256"] != _sha(CHECKPOINT)
        or protocol["source_manifest_sha256"] != _source_manifest_digest()
        or protocol["pyscf_version"] != pyscf.__version__
    ):
        raise RuntimeError("Frozen benchmark source/runtime/checkpoint drifted.")
    if (HERE / "results.json").exists():
        raise FileExistsError("Results already exist; use a new run directory.")
    cases = json.loads((PANEL / "protocol.json").read_text())["cases"]
    dielectric = route2_solvent_spec("water").descriptors.dielectric
    model = MACEPolarTorchGraphAdapter(
        build_official_mace_polar_1_m_adapter(
            device="cpu", checkpoint_path=CHECKPOINT, torch_graph=True
        )
    )
    rows = []
    output = {
        "protocol_sha256": _sha(HERE / "protocol.json"),
        "case_count": len(cases),
        "rows": rows,
        "fit_performed": False,
        "experimental_accuracy_tested": False,
    }
    for case in cases:
        name = case["name"]
        print(name, flush=True)
        row = {"name": name, "methods": {}, "rotation": {}, "forces": {}, "audits": {}}
        rows.append(row)
        try:
            atoms = _atoms(case)
            symbols = tuple(atoms.get_chemical_symbols())
            positions = torch.tensor(
                atoms.positions, dtype=torch.float64, requires_grad=True
            )
            vacuum, learned = model.energy_source_torch(atoms, positions)
            raw = learned.detach().cpu().numpy()
            if abs(float(raw[:, 0].sum())) > 1.0e-8:
                raise RuntimeError("Frozen source violates neutral charge contract.")
            cds = TorchLegacySMDCDS(symbols, "water", device="cpu")
            cds_energy = cds.energy_torch(positions)
            row["vacuum_eV"] = float(vacuum.detach())
            row["cds_eV"] = float(cds_energy.detach())
            row["source_charge_e"] = float(raw[:, 0].sum())
            baseline_path = PANEL / name / "result.json"
            baseline = json.loads(baseline_path.read_text())
            if "torch" in baseline:
                terms = baseline["torch"]["component_energies_eV"]
                if (
                    abs(row["vacuum_eV"] - terms["vacuum"]) > 1.0e-6
                    or abs(row["cds_eV"] - terms["cds"]) > 1.0e-6
                ):
                    raise RuntimeError("Frozen v3 vacuum/CDS comparison drifted.")
                row["v3_ddpcm_eV"] = terms["ddpcm"]
                row["v3_total_solv_eV"] = terms["ddpcm"] + terms["cds"]
                row["v3_forces_eV_per_A"] = baseline["torch"]["forces_eV_per_A"]
            else:
                row["v3_ddpcm_eV"] = None
                row["v3_total_solv_eV"] = None
                row["v3_forces_eV_per_A"] = None
                row["v3_unavailable_reason"] = baseline.get("error", {}).get(
                    "message", "not returned"
                )
            radii = smd_coulomb_radii(symbols, solvent="water")
            for method in METHODS:
                for order in ORDERS:
                    key = f"{method}-{ORDERS[order]}"
                    try:
                        result = _cpcm(atoms, raw, radii, dielectric, method, order)
                        result["total_solv_eV"] = (
                            result["polar_energy_eV"] + row["cds_eV"]
                        )
                        result["delta_v3_polar_eV"] = (
                            None
                            if row["v3_ddpcm_eV"] is None
                            else result["polar_energy_eV"] - row["v3_ddpcm_eV"]
                        )
                        row["methods"][key] = result
                    except Exception as error:
                        row["methods"][key] = {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
            if name in FORCE_CASES:
                nonpolar_gradient = (
                    torch.autograd.grad(
                        vacuum + cds_energy, positions, retain_graph=True
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                for order in (17, 29):
                    key = f"SWIG-{ORDERS[order]}"
                    try:
                        energy, polar_gradient = _swig_polar_gradient(
                            atoms, positions, learned, radii, dielectric, order
                        )
                        candidate_force = -(nonpolar_gradient + polar_gradient)
                        reference_force = row["v3_forces_eV_per_A"]
                        row["forces"][key] = {
                            "polar_energy_eV": energy,
                            "polar_gradient_eV_per_A": polar_gradient.tolist(),
                            "total_force_eV_per_A": candidate_force.tolist(),
                            "max_v3_force_delta_eV_per_A": (
                                None
                                if reference_force is None
                                else float(
                                    np.max(np.abs(candidate_force - reference_force))
                                )
                            ),
                            "translation_residual_eV_per_A": float(
                                np.max(np.abs(candidate_force.sum(axis=0)))
                            ),
                        }
                    except Exception as error:
                        row["forces"][key] = {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
            if name in ROTATION_CASES:
                rotations = Rotation.random(
                    ROTATION_COUNT,
                    random_state=np.random.default_rng(ROTATION_SEED),
                ).as_matrix()
                for method in METHODS:
                    key = f"{method}-302"
                    reference = row["methods"][key]
                    if "polar_energy_eV" not in reference:
                        row["rotation"][key] = {"error": "central evaluation failed"}
                        continue
                    rotated_energies = []
                    for rotation in rotations:
                        moved = atoms.copy()
                        moved.positions = atoms.positions @ rotation.T
                        moved_raw = _rotated_source(raw, rotation)
                        rotated_energies.append(
                            _cpcm(moved, moved_raw, radii, dielectric, method, 29)[
                                "polar_energy_eV"
                            ]
                        )
                    row["rotation"][key] = {
                        "rotated_polar_energies_eV": rotated_energies,
                        "max_abs_delta_eV": float(
                            np.max(
                                np.abs(
                                    np.asarray(rotated_energies)
                                    - reference["polar_energy_eV"]
                                )
                            )
                        ),
                    }
            if name in AUDIT_CASES and "SWIG-110" in row["forces"]:
                polar_gradient = row["forces"]["SWIG-110"].get(
                    "polar_gradient_eV_per_A"
                )
                if polar_gradient is not None:
                    direction = np.random.default_rng(ROTATION_SEED).normal(
                        size=atoms.positions.shape
                    )
                    direction /= np.linalg.norm(direction)
                    predicted = float(np.sum(np.asarray(polar_gradient) * direction))
                    audits = []
                    for step in AUDIT_STEPS_ANGSTROM:
                        pair = []
                        for sign in (1.0, -1.0):
                            moved = atoms.copy()
                            moved.positions += sign * step * direction
                            moved_positions = torch.tensor(
                                moved.positions, dtype=torch.float64
                            )
                            _, moved_source = model.energy_source_torch(
                                moved, moved_positions
                            )
                            pair.append(
                                _cpcm(
                                    moved,
                                    moved_source.detach().cpu().numpy(),
                                    radii,
                                    dielectric,
                                    "SWIG",
                                    17,
                                )["polar_energy_eV"]
                            )
                        finite_difference = (pair[0] - pair[1]) / (2.0 * step)
                        audits.append(
                            {
                                "step_angstrom": step,
                                "analytic_directional_eV_per_A": predicted,
                                "fd_directional_eV_per_A": finite_difference,
                                "abs_delta_eV_per_A": abs(
                                    predicted - finite_difference
                                ),
                            }
                        )
                    row["audits"]["SWIG-110"] = audits
        except Exception as error:
            row["case_error"] = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
        _write(HERE / "progress.json", output)
    _write(HERE / "results.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run()

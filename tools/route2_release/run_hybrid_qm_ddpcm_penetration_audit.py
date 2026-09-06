#!/usr/bin/env python3
"""Target-blind QM/ddPCM penetration audit for the Route-2 hybrid profile.

This runner implements the execution-independent part of the experiment frozen
in ``hybrid-qm-ddpcm-penetration-scientific-design-v1.json``.  It deliberately
contains no hydration target, CDS term, MACE score, or accuracy-based selector.

Three small subprocess-friendly modes are provided:

``validate``
    Recompute the frozen design/profile hashes and emit the exact planned cell
    matrix without importing Psi4 or pyddx.
``gas``
    Produce a gas-phase W97M-V wavefunction/density for one frozen basis and a
    content-addressed NPZ state.
``fixed-density``
    Evaluate the official Psi4 ``DdxInterface`` on that frozen gas density for
    one frozen cavity radius and report the reaction scalar, reaction-potential
    metrics, electron count, and outlying electronic charge.

The zero-shift orbital minimizer, multi-start continuation, and relaxed
occupied--virtual Hessian are intentionally *not* invented here.  Their one
allowed implementation is selected by the already-submitted Q19 Pro audit and
will be bound by a separate execution preregistration.  Keeping these modes
separate lets every expensive cell run in an isolated process with a timeout
and prevents a collapsed SCF from corrupting the rest of the matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "route2-hybrid-qm-ddpcm-penetration-audit-v1"
DESIGN_RELATIVE_PATH = Path(
    "docs/route2/preregistrations/"
    "hybrid-qm-ddpcm-penetration-scientific-design-v1.json"
)
DESIGN_SHA256 = "8607c85b38210877fb8ff2e060aec04edff3c59121fb8c06ead34fcc613d1aa5"
PROFILE_SHA256 = "c089201da915f37de02efb108720e5802b58ab55ec40f24a2a6930b1012c2287"
ALLOWED_BASIS_KEYS = ("compact", "baseline", "B1", "B2")
BASE_DISCRETIZATION = (15, 1202)
DIAGNOSTIC_DISCRETIZATIONS = ((17, 1730), (19, 2030))
DECLARED_METHOD = "W97M-V"
PSI4_METHOD = "wb97m-v"
BOHR_PER_ANGSTROM = 1.8897261254578281


@dataclass(frozen=True, slots=True)
class BasisPlan:
    """One member of the prospectively frozen nested basis sequence."""

    key: str
    psi4_name: str
    added_exponents: Mapping[str, Mapping[str, tuple[float, ...]]]

    @property
    def augmented(self) -> bool:
        return bool(self.added_exponents)


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    repository: Path
    design_path: Path
    design: Mapping[str, Any]
    profile_path: Path
    profile: Mapping[str, Any]


class AuditContractError(RuntimeError):
    """Raised when an input no longer matches the frozen scientific design."""


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise AuditContractError(f"Expected regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditContractError(message)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_frozen_inputs(
    repository: str | Path | None = None,
    *,
    expected_design_sha256: str = DESIGN_SHA256,
    expected_profile_sha256: str = PROFILE_SHA256,
) -> FrozenInputs:
    """Load and strictly validate the frozen design and geometry inputs."""

    root = (
        repository_root()
        if repository is None
        else Path(repository).expanduser().resolve(strict=True)
    )
    design_path = (root / DESIGN_RELATIVE_PATH).resolve(strict=True)
    _require(
        sha256_file(design_path) == expected_design_sha256,
        "Scientific-design preregistration bytes changed.",
    )
    design = _load_json(design_path)
    _require(design.get("schema_version") == 1, "Unexpected design schema.")
    _require(
        design.get("artifact")
        == "route2-hybrid-qm-ddpcm-penetration-scientific-design-v1",
        "Unexpected design artifact identity.",
    )
    _require(
        design.get("status")
        == "locked-before-q19-answer-and-before-terminal-execution",
        "Scientific design was not locked at the declared boundary.",
    )
    claim = design.get("claim_boundary")
    _require(isinstance(claim, dict), "Missing scientific claim boundary.")
    _require(
        claim.get("experimental_solvation_targets_read") is False,
        "Target read forbidden.",
    )
    _require(
        claim.get("cds_or_standard_state_read") is False,
        "CDS/std-state read forbidden.",
    )
    _require(
        claim.get("mace_agreement_used_for_selection") is False,
        "MACE selection forbidden.",
    )
    _require(
        claim.get("pure_mace_polar_in_scope") is False, "Pure POLAR is out of scope."
    )

    geometry = design.get("geometry")
    _require(isinstance(geometry, dict), "Missing geometry block.")
    relative_profile = geometry.get("profile_input_path")
    _require(
        isinstance(relative_profile, str) and relative_profile,
        "Missing profile input path.",
    )
    profile_path = (root / relative_profile).resolve(strict=True)
    _require(
        profile_path.is_relative_to(root),
        "Profile input must remain inside the hybrid repository.",
    )
    _require(
        geometry.get("profile_input_sha256") == expected_profile_sha256,
        "Design does not bind the expected profile input hash.",
    )
    _require(
        sha256_file(profile_path) == expected_profile_sha256,
        "Profile input bytes changed.",
    )
    profile = _load_json(profile_path)
    symbols = profile.get("symbols")
    positions = profile.get("positions_angstrom")
    radii = profile.get("radii_angstrom")
    _require(isinstance(symbols, list) and symbols, "Profile symbols are missing.")
    _require(
        isinstance(positions, list) and len(positions) == len(symbols),
        "Position/symbol cardinality mismatch.",
    )
    _require(
        isinstance(radii, list) and len(radii) == len(symbols),
        "Radius/symbol cardinality mismatch.",
    )
    position_array = np.asarray(positions, dtype=float)
    radius_array = np.asarray(radii, dtype=float)
    _require(
        position_array.shape == (len(symbols), 3), "Positions must have shape (N,3)."
    )
    _require(np.isfinite(position_array).all(), "Positions must be finite.")
    _require(
        np.isfinite(radius_array).all() and np.all(radius_array > 0.0),
        "Radii must be finite positive values.",
    )
    _require(
        set(symbols) == {"C", "H"}, "Frozen benzene profile must contain only C/H."
    )
    return FrozenInputs(root, design_path, design, profile_path, profile)


def basis_plans(design: Mapping[str, Any]) -> tuple[BasisPlan, ...]:
    sequence = design.get("basis_sequence")
    _require(isinstance(sequence, dict), "Missing basis sequence.")
    baseline = sequence.get("baseline")
    compact = sequence.get("compact_control")
    _require(compact == "def2-TZVP", "Unexpected compact control basis.")
    _require(baseline == "def2-TZVPD", "Unexpected baseline basis.")

    raw_layers: dict[str, dict[str, dict[str, float]]] = {}
    for key in ("B1", "B2"):
        raw = sequence.get(f"{key}_added_exponents")
        _require(
            isinstance(raw, dict) and set(raw) == {"C", "H"},
            f"Missing {key} exponents.",
        )
        normalized: dict[str, dict[str, float]] = {}
        for element in ("C", "H"):
            element_raw = raw[element]
            _require(
                isinstance(element_raw, dict) and set(element_raw) == {"s", "p"},
                f"{key}/{element} must contain exactly s,p exponents.",
            )
            normalized[element] = {}
            for angular in ("s", "p"):
                exponent = float(element_raw[angular])
                _require(
                    np.isfinite(exponent) and exponent > 0.0,
                    "Basis exponents must be positive finite values.",
                )
                normalized[element][angular] = exponent
        raw_layers[key] = normalized

    b1 = {
        element: {
            angular: (raw_layers["B1"][element][angular],) for angular in ("s", "p")
        }
        for element in ("C", "H")
    }
    # The second diagnostic member is nested by construction: B2 = B1 plus
    # the second prescribed diffuse primitive, not baseline plus B2 alone.
    b2 = {
        element: {
            angular: (
                raw_layers["B1"][element][angular],
                raw_layers["B2"][element][angular],
            )
            for angular in ("s", "p")
        }
        for element in ("C", "H")
    }
    result = (
        BasisPlan("compact", str(compact), {}),
        BasisPlan("baseline", str(baseline), {}),
        BasisPlan("B1", "HYBRID_QM_DDPCM_B1", b1),
        BasisPlan("B2", "HYBRID_QM_DDPCM_B2", b2),
    )
    _require(
        tuple(plan.key for plan in result) == ALLOWED_BASIS_KEYS,
        "Basis order drifted.",
    )
    return result


def radius_scales(design: Mapping[str, Any]) -> tuple[float, ...]:
    continuum = design.get("continuum")
    _require(isinstance(continuum, dict), "Missing continuum block.")
    values = tuple(float(value) for value in continuum.get("radius_scale_sequence", ()))
    _require(
        values == (1.5, 1.3, 1.2, 1.1, 1.05, 1.025, 1.0), "Radius sequence drifted."
    )
    return values


def continuum_discretizations(
    design: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Return the one production and two frozen diagnostic discretizations."""

    continuum = design.get("continuum")
    _require(isinstance(continuum, dict), "Missing continuum block.")
    base = (int(continuum.get("lmax", -1)), int(continuum.get("n_lebedev", -1)))
    raw_refinements = continuum.get("diagnostic_refinements")
    _require(isinstance(raw_refinements, list), "Missing diagnostic refinements.")
    refinements = tuple(
        (int(value[0]), int(value[1]))
        for value in raw_refinements
        if isinstance(value, list) and len(value) == 2
    )
    _require(base == BASE_DISCRETIZATION, "Base ddPCM discretization drifted.")
    _require(
        refinements == DIAGNOSTIC_DISCRETIZATIONS,
        "Diagnostic ddPCM discretizations drifted.",
    )
    return (base, *refinements)


def planned_cells(design: Mapping[str, Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        {"basis_key": plan.key, "radius_scale": scale}
        for plan in basis_plans(design)
        for scale in radius_scales(design)
    )


def _extract_gaussian94_element_block(text: str, element: str) -> str:
    pattern = re.compile(
        rf"(?ms)^\*\*\*\*\s*\n{re.escape(element)}\s+0\s*\n(.*?)^\*\*\*\*\s*$"
    )
    match = pattern.search(text)
    if match is None:
        raise AuditContractError(
            f"Element {element} is absent from baseline basis text."
        )
    return f"{element}     0\n{match.group(1).rstrip()}\n"


def augmented_basis_text(baseline_text: str, plan: BasisPlan) -> str:
    """Return exact C/H Gaussian94 text with the frozen nested shells."""

    _require(plan.augmented, "Only B1/B2 require generated basis text.")
    blocks: list[str] = ["spherical\n\n"]
    for element in ("H", "C"):
        blocks.append("****\n")
        blocks.append(_extract_gaussian94_element_block(baseline_text, element))
        for angular in ("s", "p"):
            exponents = tuple(plan.added_exponents[element][angular])
            _require(exponents, "Augmented shell list cannot be empty.")
            for exponent in exponents:
                blocks.append(
                    f"{angular.upper()}   1   1.00\n"
                    f"  {float(exponent):.17g}  1.0000000\n"
                )
    blocks.append("****\n")
    result = "".join(blocks)
    for element in ("C", "H"):
        for angular in ("s", "p"):
            for exponent in plan.added_exponents[element][angular]:
                _require(
                    f"{float(exponent):.17g}" in result,
                    "Augmented exponent serialization failed.",
                )
    return result


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_record() -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "psi4", "pyddx", "qcelemental")
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``np.savez`` appends .npz to string paths; an exclusive binary handle
    # preserves the exact requested name and prevents evidence replacement.
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _import_psi4() -> tuple[Any, Any]:
    try:
        import psi4
        from psi4.driver.procrouting.solvent import ddx
    except ImportError as exc:  # pragma: no cover - optional runtime gate
        raise AuditContractError(
            "Psi4 1.11 with pyddx 0.8.0 is required for this execution mode."
        ) from exc
    _require(str(psi4.__version__) == "1.11", "The frozen audit requires Psi4 1.11.")
    _require(
        _package_version("pyddx") == "0.8.0", "The frozen audit requires pyddx 0.8.0."
    )
    return psi4, ddx


def _molecule(psi4: Any, inputs: FrozenInputs) -> Any:
    geometry = inputs.design["geometry"]
    lines = [f"{int(geometry['charge'])} {int(geometry['multiplicity'])}"]
    for symbol, position in zip(
        inputs.profile["symbols"], inputs.profile["positions_angstrom"], strict=True
    ):
        x, y, z = (float(value) for value in position)
        lines.append(f"{symbol} {x:.15f} {y:.15f} {z:.15f}")
    lines.extend(("symmetry c1", "no_reorient", "no_com", "units angstrom"))
    return psi4.geometry("\n".join(lines))


def _baseline_basis_file(psi4: Any) -> Path:
    path = Path(psi4.core.get_datadir()) / "basis" / "def2-tzvpd.gbs"
    return path.resolve(strict=True)


def register_or_select_basis(
    psi4: Any, mol: Any, plan: BasisPlan
) -> tuple[Any, dict[str, object]]:
    if plan.augmented:
        baseline_path = _baseline_basis_file(psi4)
        text = augmented_basis_text(baseline_path.read_text(encoding="utf-8"), plan)
        psi4.basis_helper(text, name=plan.psi4_name, set_option=False)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    else:
        baseline_path = None
        text_sha = None
    basis = psi4.core.BasisSet.build(mol, "BASIS", plan.psi4_name)
    record = {
        "key": plan.key,
        "psi4_name": plan.psi4_name,
        "nbf": int(basis.nbf()),
        "nshell": int(basis.nshell()),
        "augmented": plan.augmented,
        "generated_basis_text_sha256": text_sha,
        "installed_baseline_basis_path": str(_baseline_basis_file(psi4)),
        "installed_baseline_basis_sha256": sha256_file(_baseline_basis_file(psi4)),
    }
    return basis, record


def _psi4_options(inputs: FrozenInputs, plan: BasisPlan) -> dict[str, object]:
    electronic = inputs.design["electronic_model"]
    return {
        "basis": plan.psi4_name,
        "reference": str(electronic["reference"]).lower(),
        "scf_type": str(electronic["integrals"]).lower(),
        "dft_radial_points": int(electronic["xc_radial_points"]),
        "dft_spherical_points": int(electronic["xc_spherical_points"]),
        "s_orthogonalization": "canonical",
        "s_tolerance": float(electronic["canonical_orthogonalization_cutoff"]),
        "e_convergence": 1.0e-10,
        "d_convergence": 1.0e-8,
        "maxiter": 250,
        "fail_on_maxiter": True,
    }


def _ddx_options(
    inputs: FrozenInputs,
    radius_scale: float,
    *,
    lmax: int,
    n_lebedev: int,
) -> dict[str, object]:
    continuum = inputs.design["continuum"]
    _require(
        (lmax, n_lebedev) in continuum_discretizations(inputs.design),
        "ddPCM discretization was not preregistered.",
    )
    radii = np.asarray(inputs.profile["radii_angstrom"], dtype=float) * radius_scale
    return {
        "ddx_model": str(continuum["model"]),
        "ddx_solvent_epsilon": float(continuum["epsilon"]),
        "ddx_radii": radii.tolist(),
        "ddx_lmax": int(lmax),
        "ddx_n_lebedev": int(n_lebedev),
        "ddx_eta": float(continuum["eta"]),
        "ddx_shift": float(continuum["shift"]),
        "ddx_solvation_convergence": float(continuum["solver_residual_max"]),
        "ddx_solute_spherical_points": int(
            inputs.design["electronic_model"]["xc_spherical_points"]
        ),
        "ddx_solute_radial_points": int(
            inputs.design["electronic_model"]["xc_radial_points"]
        ),
        "ddx_incore": True,
        "ddx_fmm": False,
    }


def _find_plan(inputs: FrozenInputs, key: str) -> BasisPlan:
    by_key = {plan.key: plan for plan in basis_plans(inputs.design)}
    try:
        return by_key[key]
    except KeyError as exc:
        raise AuditContractError(f"Unknown basis key {key!r}.") from exc


def electron_count(total_density: np.ndarray, overlap: np.ndarray) -> float:
    density = np.asarray(total_density, dtype=float)
    metric = np.asarray(overlap, dtype=float)
    _require(density.ndim == metric.ndim == 2, "Density/overlap must be matrices.")
    _require(density.shape == metric.shape, "Density/overlap shape mismatch.")
    _require(
        np.isfinite(density).all() and np.isfinite(metric).all(),
        "Nonfinite density/overlap.",
    )
    return float(np.einsum("ij,ji->", density, metric))


def orthonormal_operator_metrics(
    operator: np.ndarray,
    overlap: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, float | int]:
    """Report basis-comparable metrics for an AO covariant operator matrix."""

    matrix = np.asarray(operator, dtype=float)
    metric = np.asarray(overlap, dtype=float)
    _require(
        matrix.shape == metric.shape and matrix.ndim == 2, "Operator/overlap mismatch."
    )
    _require(np.allclose(metric, metric.T, atol=1.0e-12), "Overlap must be symmetric.")
    orthonormal, retained_values = orthonormal_operator_matrix(
        matrix, metric, cutoff=cutoff
    )
    singular = np.linalg.svd(orthonormal, compute_uv=False)
    return {
        "retained_rank": int(retained_values.size),
        "overlap_min_retained": float(retained_values.min()),
        "frobenius": float(np.linalg.norm(orthonormal)),
        "spectral": float(singular[0]),
        "min_eigenvalue_symmetric_part": float(
            np.linalg.eigvalsh(0.5 * (orthonormal + orthonormal.T)).min()
        ),
        "max_eigenvalue_symmetric_part": float(
            np.linalg.eigvalsh(0.5 * (orthonormal + orthonormal.T)).max()
        ),
    }


def orthonormal_operator_matrix(
    operator: np.ndarray,
    overlap: np.ndarray,
    *,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one AO covariant operator into the canonical orthonormal basis."""

    matrix = np.asarray(operator, dtype=float)
    metric = np.asarray(overlap, dtype=float)
    _require(
        matrix.shape == metric.shape and matrix.ndim == 2,
        "Operator/overlap mismatch.",
    )
    _require(
        np.isfinite(matrix).all() and np.isfinite(metric).all(),
        "Operator/overlap must be finite.",
    )
    _require(np.allclose(metric, metric.T, atol=1.0e-12), "Overlap must be symmetric.")
    values, vectors = np.linalg.eigh(metric)
    keep = values > cutoff
    _require(np.any(keep), "Canonical orthogonalization removed the full AO space.")
    x = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    return x.T @ matrix @ x, values[keep]


def fixed_density_refinement_metrics(
    *,
    base_energy_hartree: float,
    base_operator: np.ndarray,
    refined_energy_hartree: float,
    refined_operator: np.ndarray,
    overlap: np.ndarray,
    cutoff: float,
) -> dict[str, float | bool]:
    """Evaluate the frozen Q18 base-versus-highest refinement gates."""

    base_orth, base_values = orthonormal_operator_matrix(
        base_operator, overlap, cutoff=cutoff
    )
    refined_orth, refined_values = orthonormal_operator_matrix(
        refined_operator, overlap, cutoff=cutoff
    )
    _require(
        base_values.shape == refined_values.shape
        and np.array_equal(base_values, refined_values),
        "Canonical overlap rank or spectrum changed between refinements.",
    )
    energy_difference = abs(float(base_energy_hartree) - float(refined_energy_hartree))
    operator_difference = float(np.linalg.norm(base_orth - refined_orth, ord=2))
    refined_norm = float(np.linalg.norm(refined_orth, ord=2))
    relative_operator_difference = operator_difference / max(1.0, refined_norm)
    return {
        "energy_absolute_difference_hartree": energy_difference,
        "operator_spectral_difference_hartree": operator_difference,
        "refined_operator_spectral_hartree": refined_norm,
        "operator_relative_difference": relative_operator_difference,
        "energy_gate_pass": energy_difference <= 1.0e-6,
        "operator_gate_pass": relative_operator_difference <= 1.0e-4,
    }


def outlying_charge_from_points(
    density: np.ndarray,
    points_bohr: np.ndarray,
    weights_bohr3: np.ndarray,
    centers_bohr: np.ndarray,
    radii_bohr: np.ndarray,
) -> dict[str, float]:
    rho = np.asarray(density, dtype=float).reshape(-1)
    points = np.asarray(points_bohr, dtype=float)
    weights = np.asarray(weights_bohr3, dtype=float).reshape(-1)
    centers = np.asarray(centers_bohr, dtype=float)
    radii = np.asarray(radii_bohr, dtype=float).reshape(-1)
    _require(points.shape == (rho.size, 3), "Point/density shape mismatch.")
    _require(weights.shape == rho.shape, "Weight/density shape mismatch.")
    _require(centers.ndim == 2 and centers.shape[1] == 3, "Center shape mismatch.")
    _require(radii.shape == (centers.shape[0],), "Radius/center shape mismatch.")
    _require(
        np.isfinite(rho).all()
        and np.isfinite(points).all()
        and np.isfinite(weights).all()
        and np.isfinite(centers).all()
        and np.isfinite(radii).all(),
        "Outlying-charge inputs must be finite.",
    )
    squared = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    inside = np.any(squared <= radii[None, :] ** 2, axis=1)
    total = float(np.dot(rho, weights))
    exterior = float(np.dot(rho[~inside], weights[~inside]))
    return {
        "grid_electron_count": total,
        "outlying_charge_electron": exterior,
        "outlying_fraction": exterior / total if total != 0.0 else float("nan"),
    }


def _grid_density_and_outlying(
    psi4: Any,
    basis: Any,
    density_alpha: np.ndarray,
    centers_bohr: np.ndarray,
    radii_bohr: np.ndarray,
) -> dict[str, float]:
    # Build only the official W97M-V quadrature object; no SCF is performed.
    # This avoids fabricating a Wavefunction instance while evaluating exactly
    # the frozen gas density on Psi4's configured DFT grid.
    from psi4.driver.procrouting import dft

    superfunctional, _ = dft.build_superfunctional(PSI4_METHOD, restricted=True)
    potential = psi4.core.VBase.build(basis, superfunctional, "RV")
    potential.initialize()
    try:
        properties = potential.properties()
        _require(
            len(properties) >= 1,
            "Psi4 DFT grid exposes no point-function evaluator.",
        )
        point_function = properties[0]
        # For restricted Kohn--Sham Psi4's RKSFunctions RHO_A is the total
        # density when pointed at Da. Passing Da+Db would double the count.
        point_function.set_pointers(psi4.core.Matrix.from_array(density_alpha))
        total = 0.0
        exterior = 0.0
        for block_index in range(int(potential.nblocks())):
            block = potential.get_block(block_index)
            point_function.compute_points(block)
            values = point_function.point_values()
            npoints = int(block.npoints())
            # Psi4 point-value buffers are allocated to the maximum block size;
            # only the first ``npoints`` entries belong to the current block.
            rho = np.asarray(values["RHO_A"], dtype=float).reshape(-1)[:npoints]
            weights = np.asarray(block.w(), dtype=float).reshape(-1)[:npoints]
            points = np.column_stack(
                (
                    np.asarray(block.x(), dtype=float).reshape(-1)[:npoints],
                    np.asarray(block.y(), dtype=float).reshape(-1)[:npoints],
                    np.asarray(block.z(), dtype=float).reshape(-1)[:npoints],
                )
            )
            current = outlying_charge_from_points(
                rho, points, weights, centers_bohr, radii_bohr
            )
            total += current["grid_electron_count"]
            exterior += current["outlying_charge_electron"]
    finally:
        potential.finalize()
    return {
        "grid_electron_count": total,
        "outlying_charge_electron": exterior,
        "outlying_fraction": exterior / total if total != 0.0 else float("nan"),
    }


def _state_runtime_assets(psi4: Any, ddx: Any) -> dict[str, object]:
    import pyddx

    paths = {
        "psi4_module": Path(psi4.__file__).resolve(strict=True),
        "psi4_ddx_wrapper": Path(ddx.__file__).resolve(strict=True),
        "pyddx_module": Path(pyddx.__file__).resolve(strict=True),
        "baseline_basis": _baseline_basis_file(psi4),
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def run_validate(inputs: FrozenInputs) -> dict[str, object]:
    plans = basis_plans(inputs.design)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "validate",
        "status": "pass-frozen-inputs-and-cell-matrix",
        "design": {
            "path": str(inputs.design_path),
            "sha256": sha256_file(inputs.design_path),
        },
        "profile": {
            "path": str(inputs.profile_path),
            "sha256": sha256_file(inputs.profile_path),
        },
        "basis_plans": [
            {
                "key": plan.key,
                "psi4_name": plan.psi4_name,
                "added_exponents": plan.added_exponents,
            }
            for plan in plans
        ],
        "radius_scales": list(radius_scales(inputs.design)),
        "planned_cells": list(planned_cells(inputs.design)),
        "planned_cell_count": len(planned_cells(inputs.design)),
        "claim_boundary": inputs.design["claim_boundary"],
        "runtime": runtime_record(),
    }


def run_gas(
    inputs: FrozenInputs,
    *,
    basis_key: str,
    state_output: Path,
    psi4_output: Path,
    threads: int,
    memory: str,
) -> dict[str, object]:
    psi4, ddx = _import_psi4()
    del ddx
    plan = _find_plan(inputs, basis_key)
    psi4.core.clean()
    psi4.set_num_threads(threads)
    psi4.set_memory(memory)
    psi4.core.set_output_file(str(psi4_output.expanduser().resolve()), False)
    mol = _molecule(psi4, inputs)
    _, basis_record = register_or_select_basis(psi4, mol, plan)
    psi4.set_options(_psi4_options(inputs, plan))
    energy, wfn = psi4.energy(PSI4_METHOD, molecule=mol, return_wfn=True)
    overlap = np.asarray(wfn.S(), dtype=float)
    da = np.asarray(wfn.Da(), dtype=float)
    db = np.asarray(wfn.Db(), dtype=float)
    total_density = da + db
    count = electron_count(total_density, overlap)
    expected_electrons = int(sum(mol.Z(index) for index in range(mol.natom()))) - int(
        inputs.design["geometry"]["charge"]
    )
    arrays = {
        "Da": da,
        "Db": db,
        "Ca": np.asarray(wfn.Ca(), dtype=float),
        "Cb": np.asarray(wfn.Cb(), dtype=float),
        "epsilon_a": np.asarray(wfn.epsilon_a(), dtype=float),
        "epsilon_b": np.asarray(wfn.epsilon_b(), dtype=float),
        "overlap": overlap,
    }
    _write_npz_exclusive(state_output, arrays)
    state_path = state_output.expanduser().resolve(strict=True)
    gates = {
        "finite_energy": bool(np.isfinite(float(energy))),
        "electron_count": abs(count - expected_electrons)
        <= float(inputs.design["stationary_gates"]["electron_count_absolute_max"]),
        "density_finite": bool(np.isfinite(total_density).all()),
    }
    _require(all(gates.values()), f"Gas-state gate failure: {gates}")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "gas",
        "status": "pass-gas-reference",
        "basis": basis_record,
        "declared_method": DECLARED_METHOD,
        "psi4_method": PSI4_METHOD,
        "energy_hartree": float(energy),
        "electron_count": count,
        "expected_electron_count": expected_electrons,
        "gates": gates,
        "state_npz": {
            "path": str(state_path),
            "bytes": state_path.stat().st_size,
            "sha256": sha256_file(state_path),
        },
        "runtime": runtime_record(),
        "runtime_assets": _state_runtime_assets(
            psi4, __import__("psi4.driver.procrouting.solvent.ddx", fromlist=["ddx"])
        ),
    }


def _load_state_npz(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve(strict=True)
    with np.load(resolved, allow_pickle=False) as archive:
        required = {"Da", "Db", "Ca", "Cb", "epsilon_a", "epsilon_b", "overlap"}
        _require(set(archive.files) == required, "Gas-state NPZ schema mismatch.")
        result = {
            name: np.asarray(archive[name], dtype=float) for name in archive.files
        }
    _require(
        all(np.isfinite(array).all() for array in result.values()),
        "Gas-state NPZ is nonfinite.",
    )
    return result


def _state_diagnostics(state: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("is_solved", "is_solved_adjoint", "x_n_iter", "s_n_iter"):
        value = getattr(state, name, None)
        if callable(value):
            value = value()
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            result[name] = value.item() if hasattr(value, "item") else value
    for name in ("x", "s", "xi", "zeta_dip", "psi", "phi"):
        value = getattr(state, name, None)
        if value is None:
            continue
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size and np.isfinite(array).all():
            result[f"{name}_shape"] = list(array.shape)
            result[f"{name}_norm"] = float(np.linalg.norm(array))
    return result


def run_fixed_density(
    inputs: FrozenInputs,
    *,
    basis_key: str,
    radius_scale: float,
    lmax: int,
    n_lebedev: int,
    state_input: Path,
    operator_output: Path,
    psi4_output: Path,
    threads: int,
    memory: str,
) -> dict[str, object]:
    psi4, ddx = _import_psi4()
    plan = _find_plan(inputs, basis_key)
    _require(
        radius_scale in radius_scales(inputs.design),
        "Radius scale was not preregistered.",
    )
    arrays = _load_state_npz(state_input)

    psi4.core.clean()
    psi4.set_num_threads(threads)
    psi4.set_memory(memory)
    psi4.core.set_output_file(str(psi4_output.expanduser().resolve()), False)
    mol = _molecule(psi4, inputs)
    basis, basis_record = register_or_select_basis(psi4, mol, plan)
    psi4.set_options(_psi4_options(inputs, plan))
    psi4.set_options(
        _ddx_options(
            inputs,
            radius_scale,
            lmax=lmax,
            n_lebedev=n_lebedev,
        )
    )
    _require(
        arrays["Da"].shape == (basis.nbf(), basis.nbf()),
        "Gas density/basis dimension mismatch.",
    )
    # The gas-state overlap is authoritative and its dimensions are already
    # bound to this basis by the density-shape check above.
    overlap = arrays["overlap"]
    total_density = arrays["Da"] + arrays["Db"]
    count = electron_count(total_density, overlap)

    interface = ddx.DdxInterface(mol, ddx.get_ddx_options(mol), basis)
    ddx_energy, ddx_potential, state = interface.get_solvation_contributions(
        psi4.core.Matrix.from_array(total_density)
    )
    potential = np.asarray(ddx_potential, dtype=float)
    _write_npz_exclusive(operator_output, {"potential_ao": potential})
    operator_path = operator_output.expanduser().resolve(strict=True)
    metrics = orthonormal_operator_metrics(
        potential,
        overlap,
        cutoff=float(
            inputs.design["electronic_model"]["canonical_orthogonalization_cutoff"]
        ),
    )

    # Reuse Psi4's exact W97M-V DFT quadrature without performing an SCF.
    centers_bohr = np.asarray(
        [[mol.x(i), mol.y(i), mol.z(i)] for i in range(mol.natom())], dtype=float
    )
    radii_bohr = (
        np.asarray(inputs.profile["radii_angstrom"], dtype=float)
        * radius_scale
        * BOHR_PER_ANGSTROM
    )
    outlying = _grid_density_and_outlying(
        psi4, basis, arrays["Da"], centers_bohr, radii_bohr
    )
    expected_electrons = int(sum(mol.Z(index) for index in range(mol.natom())))
    gates = {
        "finite_ddx_energy": bool(np.isfinite(float(ddx_energy))),
        "finite_ddx_potential": bool(np.isfinite(potential).all()),
        "density_electron_count": abs(count - expected_electrons)
        <= float(inputs.design["stationary_gates"]["electron_count_absolute_max"]),
        "ddx_primal_solved": bool(getattr(state, "is_solved", True)),
    }
    diagnostics = {
        # Q18 applies the 1e-7 electron quadrature gate to an accepted
        # stationary point and also requires a doubled grid. This Stage-1
        # frozen-gas probe records, but does not prematurely gate on, the
        # baseline-grid error.
        "baseline_grid_electron_count_error": abs(
            outlying["grid_electron_count"] - expected_electrons
        ),
        "baseline_grid_within_later_stationary_budget": abs(
            outlying["grid_electron_count"] - expected_electrons
        )
        <= float(inputs.design["stationary_gates"]["grid_electron_count_absolute_max"]),
        "doubled_grid_not_yet_run": True,
    }
    _require(all(gates.values()), f"Fixed-density gate failure: {gates}")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "fixed-density",
        "status": "pass-fixed-density-probe",
        "basis": basis_record,
        "radius_scale": radius_scale,
        "discretization": {"lmax": lmax, "n_lebedev": n_lebedev},
        "radii_angstrom": (
            np.asarray(inputs.profile["radii_angstrom"], dtype=float) * radius_scale
        ).tolist(),
        "gas_state_npz": {
            "path": str(state_input.expanduser().resolve(strict=True)),
            "sha256": sha256_file(state_input),
        },
        "density_electron_count": count,
        "ddx_energy_hartree": float(ddx_energy),
        "ddx_potential_ao": {
            "min_hartree": float(potential.min()),
            "max_hartree": float(potential.max()),
            "frobenius_hartree": float(np.linalg.norm(potential)),
        },
        "ddx_potential_npz": {
            "path": str(operator_path),
            "bytes": operator_path.stat().st_size,
            "sha256": sha256_file(operator_path),
        },
        "ddx_potential_orthonormal": metrics,
        "outlying_charge": outlying,
        "ddx_state": _state_diagnostics(state),
        "diagnostics": diagnostics,
        "gates": gates,
        "runtime": runtime_record(),
        "runtime_assets": _state_runtime_assets(psi4, ddx),
    }


def _common_record(
    inputs: FrozenInputs, mode_record: Mapping[str, object]
) -> dict[str, object]:
    runner = Path(__file__).resolve(strict=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "scientific_design": {
            "path": str(inputs.design_path),
            "sha256": sha256_file(inputs.design_path),
        },
        "profile_input": {
            "path": str(inputs.profile_path),
            "sha256": sha256_file(inputs.profile_path),
        },
        "measurement": dict(mode_record),
        "claim_boundary": {
            "experimental_solvation_targets_read": False,
            "cds_or_standard_state_read": False,
            "mace_agreement_used_for_selection": False,
            "public_capability_admitted": False,
            "terminal_orbital_stability_classified": False,
        },
    }
    payload["self_sha256"] = canonical_json_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository_root())
    subparsers = parser.add_subparsers(dest="mode", required=True)

    validate = subparsers.add_parser("validate", help="validate frozen inputs/cells")
    validate.add_argument("--output", type=Path, required=True)

    gas = subparsers.add_parser("gas", help="compute one frozen gas reference")
    gas.add_argument("--basis", choices=ALLOWED_BASIS_KEYS, required=True)
    gas.add_argument("--state-output", type=Path, required=True)
    gas.add_argument("--psi4-output", type=Path, required=True)
    gas.add_argument("--output", type=Path, required=True)
    gas.add_argument("--threads", type=int, default=8)
    gas.add_argument("--memory", default="8 GB")

    fixed = subparsers.add_parser("fixed-density", help="probe one frozen ddPCM cell")
    fixed.add_argument("--basis", choices=ALLOWED_BASIS_KEYS, required=True)
    fixed.add_argument("--radius-scale", type=float, required=True)
    fixed.add_argument("--lmax", type=int)
    fixed.add_argument("--n-lebedev", type=int)
    fixed.add_argument("--state-input", type=Path, required=True)
    fixed.add_argument("--operator-output", type=Path, required=True)
    fixed.add_argument("--psi4-output", type=Path, required=True)
    fixed.add_argument("--output", type=Path, required=True)
    fixed.add_argument("--threads", type=int, default=8)
    fixed.add_argument("--memory", default="8 GB")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = load_frozen_inputs(args.repository)
    if args.mode == "validate":
        measurement = run_validate(inputs)
    elif args.mode == "gas":
        _require(args.threads >= 1, "Thread count must be positive.")
        measurement = run_gas(
            inputs,
            basis_key=args.basis,
            state_output=args.state_output,
            psi4_output=args.psi4_output,
            threads=args.threads,
            memory=args.memory,
        )
    elif args.mode == "fixed-density":
        _require(args.threads >= 1, "Thread count must be positive.")
        _require(
            (args.lmax is None) == (args.n_lebedev is None),
            "lmax and n_lebedev must be supplied together.",
        )
        lmax, n_lebedev = (
            BASE_DISCRETIZATION if args.lmax is None else (args.lmax, args.n_lebedev)
        )
        measurement = run_fixed_density(
            inputs,
            basis_key=args.basis,
            radius_scale=args.radius_scale,
            lmax=lmax,
            n_lebedev=n_lebedev,
            state_input=args.state_input,
            operator_output=args.operator_output,
            psi4_output=args.psi4_output,
            threads=args.threads,
            memory=args.memory,
        )
    else:  # pragma: no cover - argparse contract
        raise AssertionError(args.mode)
    payload = _common_record(inputs, measurement)
    _write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "self_sha256": payload["self_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

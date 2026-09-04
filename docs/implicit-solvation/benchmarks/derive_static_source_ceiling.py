#!/usr/bin/env python3
"""Derive the V0 static-source MEP ceiling from a declared ``ΔG_solv`` tolerance.

Motivation
----------
The zero-field permanent-source gate in
``route2-v0-gfn2-molden-static-mep-acetone-v2.json`` uses the ceilings
``0.20`` (relative Frobenius) and ``0.30`` (relative max-abs). Those two
numbers were carried over unchanged from the *response*-source gate. Nothing in
the preregistration derives them from a solvation-energy tolerance, so the
route has never checked whether they correspond to the quantity it cares about.

This script derives the ceiling instead of inheriting it, using only frozen
artifacts and the standard library. It writes nothing and admits nothing: it
produces a criterion and applies it to already-recorded numbers.

Derivation
----------
The V0-RK common scalar at fixed geometry is

    G(x; v0) = ½ xᵀC⁺x + ½ (v0+Bx)ᵀQ(v0+Bx) + xᵀf ,   N x = 0 .

Let ``G*(v0) = min_x G(x; v0)`` and let ``x*`` be its stationary point. By the
envelope theorem the leading sensitivity of the *minimum value* to the
permanent source is the partial derivative at ``x*``:

    dG*/dv0 = Q (v0 + B x*) = q* ,

so a source error ``δv0`` shifts the solvation energy by

    δG* = ⟨q*, δv0⟩ + O(‖δv0‖²) .                                       (1)

This is exact to first order and involves no cavity choice beyond the one
already fixed by ``Q``. Cauchy-Schwarz then bounds it by the two norms:

    |δG*| ≤ ‖q*‖ ‖δv0‖ = ε_F ‖q*‖ ‖v0‖ ,                                (2)

where ``ε_F = ‖δv0‖/‖v0‖`` is exactly the gate's reported relative Frobenius
error. Writing the polarization energy as ``ΔG_pol = ½⟨v*, q*⟩`` and defining
the alignment factor

    κ = ‖v*‖ ‖q*‖ / |⟨v*, q*⟩|  ≥ 1     (Cauchy-Schwarz; κ = 1 iff q* ∥ v*)

gives ``‖q*‖‖v*‖ = 2κ|ΔG_pol|`` and therefore

    |δG*| ≤ 2 κ ρ ε_F |ΔG_pol| ,        ρ = ‖v0‖/‖v*‖ ,                 (3)

so for a declared tolerance ``τ`` on the solvation energy,

    ε_F^max = τ / (2 κ ρ |ΔG_pol|) .                                    (4)

``κ ≥ 1`` always and ``ρ ≥ 1`` whenever the induced surface potential screens
the permanent one (the usual case for a neutral solute). Setting ``κ = ρ = 1``
therefore yields the **most permissive ceiling the derivation can license**. A
candidate that fails at ``κ = ρ = 1`` fails for every possible alignment, with
no knowledge of ``q*`` required. That is what makes the verdict below robust.

What the protocol should have frozen
------------------------------------
Equation (1) shows the deciding quantity is the single inner product
``⟨q*, δv0⟩`` — one dot product over the 516 surface points. The frozen
artifact preserves only two scalar norms, which bound that inner product across
a factor-of-κ range but cannot evaluate it. ``--induced-charge`` accepts a
``q*`` vector and reports the exact first-order shift when it is available.

Usage
-----
    python3 derive_static_source_ceiling.py
    python3 derive_static_source_ceiling.py --tolerance 0.5
    python3 derive_static_source_ceiling.py --induced-charge q_star.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HARTREE_KCAL = 627.509474063056
NUCLEAR_CHARGE = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}

HERE = Path(__file__).resolve().parent
REPRODUCERS = HERE / "reproducers"
POINTS_NPY = (REPRODUCERS / "route2-v0-atomic-displacement-source-acetone-v1"
              / "frozen-exterior-qm-mep-points-bohr.npy")
FIELD_JSON = (REPRODUCERS / "route2-v0-mace-mdp-induced-source-acetone-v1"
              / "qm-induced-mep.json")
GEOMETRY_XYZ = (REPRODUCERS / "route2-v0-gfn2-molden-permanent-source-acetone-v1"
                / "acetone.xyz")
PROJECTION_JSON = HERE / "route2-gto-pcm-energy-projection-acetone-v1.json"
GFN2_JSON = HERE / "route2-v0-gfn2-molden-static-mep-acetone-v2.json"

# Ceilings the static-source gate inherited from the response-source gate.
INHERITED_FROBENIUS_CEILING = 0.20
INHERITED_MAX_ABS_CEILING = 0.30


def read_npy_float64(path: Path) -> tuple[list[list[float]], tuple[int, ...]]:
    """Minimal reader for a C-ordered little-endian float64 .npy (no numpy)."""
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path} is not a .npy file.")
    major = raw[6]
    if major == 1:
        header_length = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    else:
        header_length = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    header = raw[offset:offset + header_length].decode("latin-1")
    if "'<f8'" not in header or "'fortran_order': False" not in header:
        raise ValueError(f"{path} must be C-ordered float64; got {header!r}")
    shape = tuple(
        int(token)
        for token in header.split("'shape':")[1].split("(")[1].split(")")[0].split(",")
        if token.strip()
    )
    body = raw[offset + header_length:]
    flat = struct.unpack(f"<{len(body) // 8}d", body)
    if shape[-1] != 3:
        raise ValueError(f"expected (n, 3) points, got {shape}")
    return [list(flat[i * 3:i * 3 + 3]) for i in range(shape[0])], shape


def read_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = [line for line in path.read_text().splitlines()[2:] if line.strip()]
    elements = [line.split()[0] for line in lines]
    positions = [
        [float(value) * BOHR_PER_ANGSTROM for value in line.split()[1:4]]
        for line in lines
    ]
    return elements, positions


def l2(values) -> float:
    return math.sqrt(sum(value * value for value in values))


def reconstruct_static_mep(field_json: Path) -> tuple[list[float], float]:
    """Zero-field electronic MEP from ± finite-field pairs.

    ``V(+E) + V(-E) = 2 V(0) + O(E²)``, so each sign pair gives one estimate.
    The spread across (direction, step) pairs is returned as a consistency
    figure; it bounds the residual O(E²) contamination.
    """
    with field_json.open(encoding="utf-8") as handle:
        records = json.load(handle)["field_records"]

    pairs: dict[tuple[str, float], dict[int, list[float]]] = defaultdict(dict)
    for record in records:
        key = (record["direction"], record["step_au"])
        pairs[key][record["sign"]] = record["electronic_potential_hartree_per_e"]

    estimates = []
    for key in sorted(pairs):
        signs = pairs[key]
        if set(signs) != {1, -1}:
            raise ValueError(f"field pair {key} is missing a sign.")
        estimates.append(
            [(plus + minus) / 2.0 for plus, minus in zip(signs[1], signs[-1])]
        )

    count = len(estimates[0])
    consensus = [
        sum(estimate[i] for estimate in estimates) / len(estimates)
        for i in range(count)
    ]
    spread = max(
        abs(estimate[i] - consensus[i])
        for estimate in estimates
        for i in range(count)
    )
    return consensus, spread


def nuclear_potential(
    points: list[list[float]],
    elements: list[str],
    positions: list[list[float]],
) -> list[float]:
    return [
        sum(
            NUCLEAR_CHARGE[element] / math.dist(point, position)
            for element, position in zip(elements, positions)
        )
        for point in points
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tolerance", type=float, default=1.5,
        help="declared ΔG_solv tolerance τ in kcal/mol allocated to the "
             "permanent-source error alone (default: 1.5, the route's own "
             "all-record gate, i.e. the entire budget spent on this one term)",
    )
    parser.add_argument(
        "--induced-charge", type=Path, default=None,
        help="optional JSON list of the 516 converged induced surface charges "
             "q*; enables the exact first-order evaluation of equation (1)",
    )
    args = parser.parse_args()

    points, shape = read_npy_float64(POINTS_NPY)
    elements, positions = read_xyz(GEOMETRY_XYZ)
    electronic, spread = reconstruct_static_mep(FIELD_JSON)
    if len(electronic) != len(points):
        raise ValueError("point count and potential count disagree.")

    nuclear = nuclear_potential(points, elements, positions)
    v0 = [n + e for n, e in zip(nuclear, electronic)]

    with PROJECTION_JSON.open(encoding="utf-8") as handle:
        projection = json.load(handle)
    arm = projection["basis_results"]["one_radial"]
    delta_g_pol = arm["target_polarization_energy_kcal_per_mol"]
    operator = arm["operator"]

    print("== Frozen inputs ==")
    print(f"  surface points                 : {shape[0]}")
    print(f"  PCM operator surface points    : {operator['surface_point_count']}"
          f"  ({operator['operator']})")
    print(f"  ± field reconstruction spread  : {spread:.3e} hartree/e")
    print(f"  ||v0||                         : {l2(v0):.6f} hartree/e")
    print(f"     ||nuclear||={l2(nuclear):.4f}  ||electronic||={l2(electronic):.4f}"
          "   (near-cancelling)")
    print(f"  target ΔG_pol                  : {delta_g_pol:.10f} kcal/mol")

    tolerance = args.tolerance
    print(f"\n== Derived ceiling, equation (4), τ = {tolerance} kcal/mol ==")
    print("   ε_F^max = τ / (2 κ ρ |ΔG_pol|),  κ ≥ 1, ρ ≥ 1")
    for kappa in (1.0, 2.0, 5.0, 10.0):
        ceiling = tolerance / (2.0 * kappa * abs(delta_g_pol))
        note = "  <-- most permissive licensable" if kappa == 1.0 else ""
        print(f"   κ={kappa:5.1f} (ρ=1) : ε_F^max = {ceiling:.4f}{note}")

    most_permissive = tolerance / (2.0 * abs(delta_g_pol))

    print("\n== Re-adjudication of the inherited ceiling ==")
    print(f"   inherited Frobenius ceiling    : {INHERITED_FROBENIUS_CEILING:.4f}")
    print(f"   most permissive derived ceiling: {most_permissive:.4f}")
    ratio = INHERITED_FROBENIUS_CEILING / most_permissive
    print(f"   inherited / derived            : {ratio:.2f}x too LOOSE")
    admitted = 2.0 * INHERITED_FROBENIUS_CEILING * abs(delta_g_pol)
    print(f"   worst-case ΔG_pol error a source at the inherited ceiling")
    print(f"   could carry (κ=ρ=1)            : {admitted:.3f} kcal/mol")

    with GFN2_JSON.open(encoding="utf-8") as handle:
        checks = json.load(handle)["scientific_falsification"]["checks"]
    observed = checks["static_mep_relative_frobenius"]["value"]
    print("\n== The recorded GFN2 MOLDEN candidate ==")
    print(f"   observed ε_F                   : {observed:.10f}")
    print(f"   worst-case ΔG_pol error (κ=ρ=1): "
          f"{2.0 * observed * abs(delta_g_pol):.3f} kcal/mol")
    print(f"   vs derived ceiling {most_permissive:.4f}          : "
          f"{'REJECT' if observed > most_permissive else 'admit'}"
          f" (by {observed / most_permissive:.2f}x)")
    print("   Rejection holds for every κ ≥ 1 and ρ ≥ 1, so it does not depend")
    print("   on the unknown alignment of the error with q*.")

    if args.induced_charge is not None:
        with args.induced_charge.open(encoding="utf-8") as handle:
            q_star = json.load(handle)
        if len(q_star) != len(v0):
            raise ValueError("q* length must match the surface point count.")
        print("\n== Exact first-order evaluation, equation (1) ==")
        print(f"   ||q*||                         : {l2(q_star):.6f} e")
        kappa = l2(v0) * l2(q_star) / abs(
            sum(a * b for a, b in zip(v0, q_star))
        )
        print(f"   κ from supplied q*             : {kappa:.3f}")
        print(f"   ε_F^max at κ, ρ=1              : "
              f"{tolerance / (2.0 * kappa * abs(delta_g_pol)):.4f}")
        print("   Supply δv0 as well to evaluate ⟨q*, δv0⟩ directly.")
    else:
        print("\n   Supply --induced-charge to replace the κ range with the one")
        print("   inner product that actually decides the question.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

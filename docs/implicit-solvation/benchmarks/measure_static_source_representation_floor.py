#!/usr/bin/env python3
"""Measure the representation floor of the V0 static-source MEP gate.

The gate in ``route2-v0-gfn2-molden-static-mep-acetone-v2.json`` compares an
atom-centred source *inside* the cavity against an all-electron QM MEP
evaluated *on* that cavity. This script asks a question the gate never asked:
what is the smallest error any such source can achieve?

It fits the best possible atom-centred point-multipole source directly to the
frozen QM reference by linear least squares and reports the residual. That
residual is a floor: no source model, MLIP or QM, can score below it in that
representation.

Only the standard library is used. Nothing is written, no model is run, and no
threshold is selected.

Usage:  python3 measure_static_source_representation_floor.py
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HERE = Path(__file__).resolve().parent
REPRO = HERE / "reproducers"
POINTS = REPRO / "route2-v0-atomic-displacement-source-acetone-v1" / "frozen-exterior-qm-mep-points-bohr.npy"
GEOM = REPRO / "route2-v0-gfn2-molden-permanent-source-acetone-v1" / "acetone.xyz"
GATE = HERE / "route2-v0-gfn2-molden-static-mep-acetone-v2.json"

# Recorded candidates on this exact surface, for context only.
RECORDED = {
    "GFN2 MOLDEN (full AO density, rejected)": 0.2199987983,
    "MACE-EF energy-conjugate": 0.2777210207,
    "MACE-EF auxiliary density": 0.2954775346,
    "MACE-POLAR-1-M l<=1": 0.3224217790,
}
# From ROUTE2_V0_STATIC_SOURCE_CEILING_DERIVATION.md, kappa = rho = 1.
CEILING_TAU_1_5 = 0.1129
CEILING_TAU_0_5 = 0.0376


def read_points(path: Path) -> list[list[float]]:
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path} is not a .npy file.")
    header_length = struct.unpack("<H", raw[8:10])[0]
    body = raw[10 + header_length:]
    flat = struct.unpack(f"<{len(body) // 8}d", body)
    return [list(flat[i * 3:i * 3 + 3]) for i in range(len(flat) // 3)]


def read_geometry(path: Path) -> list[list[float]]:
    lines = [line for line in path.read_text().splitlines()[2:] if line.strip()]
    return [
        [float(value) * BOHR_PER_ANGSTROM for value in line.split()[1:4]]
        for line in lines
    ]


def basis(point: list[float], centre: list[float], lmax: int) -> list[float]:
    dx, dy, dz = (point[i] - centre[i] for i in range(3))
    r2 = dx * dx + dy * dy + dz * dz
    r = math.sqrt(r2)
    r3, r5 = r * r2, r * r2 * r2
    functions = [1.0 / r, dx / r3, dy / r3, dz / r3]
    if lmax >= 2:
        functions += [
            dx * dy / r5, dy * dz / r5, dz * dx / r5,
            (dx * dx - dy * dy) / r5, (3.0 * dz * dz - r2) / r5,
        ]
    return functions


def solve(augmented: list[list[float]], n: int) -> list[float]:
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-13:
            raise ValueError("singular normal equations")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]
    return [augmented[i][n] / augmented[i][i] for i in range(n)]


def norm(values) -> float:
    return math.sqrt(sum(v * v for v in values))


def main() -> int:
    points = read_points(POINTS)
    centres = read_geometry(GEOM)
    with GATE.open(encoding="utf-8") as handle:
        reference = json.load(handle)["raw_static_mep"]["qm_all_electron_hartree_per_e"]
    n_points, n_atoms = len(points), len(centres)
    ref_l2, ref_inf = norm(reference), max(abs(v) for v in reference)

    def fit(lmax: int, total_charge: float | None):
        per = 4 if lmax == 1 else 9
        design = [[c for a in centres for c in basis(s, a, lmax)] for s in points]
        n = len(design[0])
        ata = [[sum(design[k][i] * design[k][j] for k in range(n_points))
                for j in range(n)] for i in range(n)]
        atb = [sum(design[k][i] * reference[k] for k in range(n_points)) for i in range(n)]
        if total_charge is None:
            coefficients = solve([ata[i][:] + [atb[i]] for i in range(n)], n)
        else:
            kkt = [ata[i][:] + [1.0 if i % per == 0 else 0.0] for i in range(n)]
            kkt.append([1.0 if j % per == 0 else 0.0 for j in range(n)] + [0.0])
            rhs = atb + [total_charge]
            coefficients = solve([kkt[i][:] + [rhs[i]] for i in range(n + 1)], n + 1)[:n]
        fitted = [sum(design[k][i] * coefficients[i] for i in range(n))
                  for k in range(n_points)]
        residual = [a - b for a, b in zip(fitted, reference)]
        charges = [coefficients[per * a] for a in range(n_atoms)]
        return (norm(residual) / ref_l2,
                max(abs(v) for v in residual) / ref_inf,
                sum(charges),
                max(abs(c) for c in charges))

    print(f"surface points {n_points}, atoms {n_atoms}, ||v_QM|| = {ref_l2:.6f} hartree/e\n")
    print("Best achievable atom-centred source, fitted to the QM reference itself:")
    print(f"  {'model':28s} {'eps_F':>10s} {'eps_maxabs':>11s} {'sum q':>9s} {'max|q|':>8s}")
    for lmax in (1, 2):
        for charge, label in ((None, "free"), (0.0, "charge-neutral")):
            e_f, e_m, q_sum, q_max = fit(lmax, charge)
            print(f"  l<={lmax} {label:22s} {e_f:10.6f} {e_m:11.6f}"
                  f" {q_sum:+9.4f} {q_max:8.3f}")

    print("\nImposed enclosed charge scan (l<=1):")
    best = None
    for charge in (0.0, 0.01, 0.02, 0.03, 0.04, 0.0437, 0.05, 0.06, 0.08, 0.10):
        e_f, _, _, _ = fit(1, charge)
        if best is None or e_f < best[1]:
            best = (charge, e_f)
        print(f"  Q = {charge:+.4f} e   eps_F = {e_f:.6f}")
    print(f"  minimum near Q = {best[0]:+.4f} e")

    print("\nContext:")
    print(f"  {'derived ceiling (tau=1.5)':40s} {CEILING_TAU_1_5:.4f}")
    print(f"  {'derived ceiling (tau=0.5)':40s} {CEILING_TAU_0_5:.4f}")
    for name, value in RECORDED.items():
        print(f"  {name:40s} {value:.4f}")

    floor, _, _, _ = fit(1, 0.0)
    print(f"\n  charge-neutral l<=1 representation floor  {floor:.4f}")
    if floor > CEILING_TAU_1_5:
        print("  -> The floor EXCEEDS the ceiling. In this representation the gate")
        print("     cannot be passed by any source, MLIP or QM.")
        print("     NOTE: the released degree of freedom is a near-uniform offset,")
        print("     NOT enclosed charge -- cos(g, constant) = 0.996 on this surface")
        print("     and a free constant alone recovers 99.7% of the gap. See")
        print("     ROUTE2_V0_STATIC_SOURCE_CEILING_DERIVATION.md section 9.2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

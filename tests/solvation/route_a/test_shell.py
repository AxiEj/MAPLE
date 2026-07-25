from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pytest

from .conftest import DOCS_DIR, PROJECT_ROOT, load_json

PROTO_PATH = DOCS_DIR / "protocol-v1.json"
RADIUS_PATH = PROJECT_ROOT / "docs" / "solvation" / "route-a" / "radius-profiles" / "bondi-mantina-v1.json"

def _load_protocol():
    return load_json(PROTO_PATH)


def _kb_ev_per_k():
    return 8.617333262145e-5


def _wall_stiffness_kbt_eVA2(protocol: dict) -> float:
    wall = protocol["shell"]["wall"]
    width = float(wall["buffer_width_A"])
    kbt_scale = float(wall["buffer_height_kBT"])
    beta_ev_inv = 1.0 / (_kb_ev_per_k() * float(protocol["standard_state"]["temperature_k"]))
    k_bt_ev = 1.0 / beta_ev_inv
    return 2.0 * kbt_scale * k_bt_ev / (width ** 2)


def _wall_potential(delta_A: float, protocol: dict) -> float:
    wall = protocol["shell"]["wall"]
    x = max(0.0, float(delta_A))
    if x == 0.0:
        return 0.0
    k = _wall_stiffness_kbt_eVA2(protocol)
    return 0.5 * k * x * x


def _signed_distance_union(point: np.ndarray, solute_pos: np.ndarray, solute_radii: np.ndarray) -> float:
    d = np.linalg.norm(point[None, :] - solute_pos, axis=1) - solute_radii
    return float(np.min(d))


def _water_occupancy(
    positions: np.ndarray,
    symbols: list[str],
    solute_atom_idx: Sequence[int],
    water_groups: Sequence[tuple[int, int, int]],
    radii_by_symbol: dict[str, float],
    lambda_s: float,
) -> list[int]:
    # complete-molecule retained occupancy policy for route-a prototype:
    # water O decides membership; if retained, keep all atoms in that group.
    solute_pos = np.asarray([positions[i] for i in solute_atom_idx], dtype=float)
    solute_rad = np.asarray([radii_by_symbol[symbols[i]] for i in solute_atom_idx], dtype=float)

    retained: list[int] = []
    for i_o, i_h1, i_h2 in water_groups:
        # oxygen-only membership for each water molecule, with all-solute-atom distance metric
        d = _signed_distance_union(positions[i_o], solute_pos, solute_rad)
        if d <= lambda_s:
            retained.extend([i_o, i_h1, i_h2])

    return retained


def test_union_signed_distance_is_inside_outside_boundary_inclusive_and_counterexample_exists():
    protocol = _load_protocol()
    R = load_json(RADIUS_PATH)["radii"]

    solute_pos = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    solute_r = np.array([R["O"], R["O"]], dtype=float)

    center = np.array([solute_r[0], 0.0, 0.0])
    near = np.array([solute_r[0] + 0.48, 0.0, 0.0])
    assert np.isclose(_signed_distance_union(center, solute_pos, solute_r), 0.0)
    assert _signed_distance_union(near, solute_pos, solute_r) > 0.0

    lambda_s = 0.5
    query = np.array([-solute_r[0] - lambda_s, 0.0, 0.0])
    assert _signed_distance_union(query, solute_pos, solute_r) == pytest.approx(lambda_s, rel=0, abs=0.0)

    eps = 1e-8
    assert _signed_distance_union(query + eps * np.array([1.0, 0.0, 0.0]), solute_pos, solute_r) <= lambda_s + 1e-8
    assert _signed_distance_union(query - eps * np.array([1.0, 0.0, 0.0]), solute_pos, solute_r) > lambda_s - 1e-8

    # oxygen-only union can miss a close non-oxygen atom; all-atom union catches it.
    symbols = ["O", "C", "O", "H", "H", "H", "O", "H", "H"]
    radii = {"H": 1.1, "C": 1.6, "O": R["O"]}
    pos = np.array(
        [
            [0.0, 0.0, 0.0],  # O (far from probe)
            [2.6, 0.0, 0.0],  # C (close to probe)
            [6.0, 0.0, 0.0],  # O (distant)
            [0.3, 0.8, 0.0],
            [0.3, -0.8, 0.0],
            [6.2, 0.7, 0.0],
            [3.6, 0.0, 0.0],  # target water O
            [3.7, 0.3, 0.0],
            [3.8, -0.3, 0.0],
        ],
        dtype=float,
    )

    oxy_groups = [(6, 7, 8)]
    lambda_probe = 0.6

    all_atoms = [0, 1, 2]
    oxygen_only = [0, 2]

    all_atom_d = _signed_distance_union(pos[6], pos[all_atoms], np.array([radii[symbols[i]] for i in all_atoms]))
    oxygen_only_d = _signed_distance_union(pos[6], pos[oxygen_only], np.array([radii[symbols[i]] for i in oxygen_only]))

    assert all_atom_d <= lambda_probe
    assert oxygen_only_d > lambda_probe

    retained_all = _water_occupancy(pos, symbols, all_atoms, oxy_groups, radii, lambda_probe)
    retained_oxygen = _water_occupancy(pos, symbols, oxygen_only, oxy_groups, radii, lambda_probe)
    assert retained_all == [6, 7, 8]
    assert retained_oxygen == []


def test_complete_water_occupancy_is_oxygen_decisive_and_hydrogens_insensitive():
    protocol = _load_protocol()
    assert protocol["shell"]["complete_molecule_retained"] is True

    # synthetic three-atom water fragment, no periodicity assumed
    symbols = ["O", "C", "H", "H"]
    radii = load_json(RADIUS_PATH)["radii"]
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.52, 0.0, 0.0],  # shell O
            [2.8, 0.2, 0.0],
            [2.8, -0.2, 0.0],
        ],
        dtype=float,
    )

    lambda_s = 1.0
    solute_atoms = [0]
    groups = [(1, 2, 3)]
    retained = _water_occupancy(pos, symbols, solute_atoms, groups, radii, lambda_s)
    assert retained == [1, 2, 3]

    shifted = pos.copy()
    shifted[2, :] += np.array([0.25, 0.0, 0.0])
    shifted[3, :] += np.array([-0.31, 0.0, 0.0])

    shifted_retained = _water_occupancy(shifted, symbols, solute_atoms, groups, radii, lambda_s)
    assert shifted_retained == retained


@pytest.mark.parametrize("delta", [(-1e-8), (0.0), (1e-8), (0.5)])
def test_flat_bottom_harmonic_wall_is_zero_inside_continuous_and_c1_centered(delta):
    protocol = _load_protocol()
    wall = protocol["shell"]["wall"]
    assert wall["type"] == "flat_bottom_harmonic_buffer"
    assert wall["differentiable"] is True

    eps = 1e-8
    left = _wall_potential(-eps, protocol)
    right = _wall_potential(+eps, protocol)
    assert left == pytest.approx(0.0, abs=0.0)
    assert right == pytest.approx(0.0, abs=2e-15)

    center = _wall_potential(delta, protocol)
    if delta <= 0.0:
        dd = 1e-9
        force_fd = -(_wall_potential(delta, protocol) - _wall_potential(delta - dd, protocol)) / dd
    else:
        dd = 4e-8
        force_fd = -(_wall_potential(delta + dd, protocol) - _wall_potential(delta - dd, protocol)) / (2.0 * dd)

    if delta <= 0.0:
        assert force_fd == pytest.approx(0.0, abs=5e-10)
    else:
        expected = -_wall_stiffness_kbt_eVA2(protocol) * delta
        assert force_fd == pytest.approx(expected, rel=5e-2, abs=2e-8)



def test_wall_at_buffer_width_hits_10_kBT_stays_finite_and_remains_harmonic_afterwards():
    protocol = _load_protocol()
    wall = protocol["shell"]["wall"]
    kbt = float(_kb_ev_per_k() * float(protocol["standard_state"]["temperature_k"]))
    width = float(wall["buffer_width_A"])

    v_width = _wall_potential(width, protocol)
    assert v_width == pytest.approx(10.0 * kbt, rel=1e-12, abs=1e-12)
    assert np.isfinite(v_width)

    k = _wall_stiffness_kbt_eVA2(protocol)
    expected_k = 2.0 * (10.0 * kbt) / (width ** 2)
    assert k == pytest.approx(expected_k, rel=1e-12)

    u2w = _wall_potential(2.0 * width, protocol)
    assert u2w == pytest.approx(0.5 * expected_k * (2.0 * width) ** 2, rel=1e-12)
    assert np.isfinite(u2w)


def _two_sphere_union_volume_area(z_grid: np.ndarray, r: float, d: float) -> np.ndarray:
    areas = []
    for z in z_grid:
        rz2 = r * r - z * z
        if rz2 <= 0.0:
            areas.append(0.0)
            continue

        rr = np.sqrt(rz2)
        # area of one circle cross-section
        a1 = np.pi * rr * rr

        # with symmetry, this reduces to intersection of circles at equal z-plane radii
        # sphere2 center offset along x by d
        if d <= 0.0:
            a_union = a1
        elif d >= 2.0 * rr:
            a_union = 2.0 * a1
        else:
            r0 = rr
            r1 = rr
            dxy = np.abs(d)
            alpha = (r0 * r0 + dxy * dxy - r1 * r1) / (2.0 * r0 * dxy)
            beta = (r1 * r1 + dxy * dxy - r0 * r0) / (2.0 * r1 * dxy)
            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)
            t0 = 2.0 * np.arccos(alpha)
            t1 = 2.0 * np.arccos(beta)
            inter = 0.5 * r0 * r0 * (t0 - np.sin(t0)) + 0.5 * r1 * r1 * (t1 - np.sin(t1))
            a_union = 2.0 * a1 - inter

        areas.append(a_union)

    return np.asarray(areas, dtype=float)


def _union_volume_trapezoid(r: float, d: float, n: int) -> float:
    zmax = r
    z = np.linspace(-zmax, zmax, n)
    area = _two_sphere_union_volume_area(z, r, d)
    return float(np.trapezoid(area, z))


def _equal_sphere_overlap_volume(r: float, d: float) -> float:
    # Intersection of equal spheres; returns overlap only when |d| < 2r.
    d = float(np.abs(d))
    if d >= 2.0 * r:
        return 0.0
    if d <= 0.0:
        return 4.0 / 3.0 * np.pi * r ** 3
    return np.pi * (4.0 * r + d) * (2.0 * r - d) ** 2 / 12.0


def _single_sphere_soft_veff_closed_form(
    flat_radius_A: float,
    beta_eV_inv: float,
    k_eV_per_A2: float,
) -> float:
    """Integral of exp[-beta*0.5*k*max(r-a,0)^2] over R^3."""
    a = float(flat_radius_A)
    alpha = 0.5 * beta_eV_inv * k_eV_per_A2
    hard = a ** 3 / 3.0
    tail = (
        a * a * np.sqrt(np.pi) / (2.0 * np.sqrt(alpha))
        + a / alpha
        + np.sqrt(np.pi) / (4.0 * alpha ** 1.5)
    )
    return 4.0 * np.pi * (hard + tail)


def _two_sphere_soft_veff_cylindrical(
    vdw_radius_A: float,
    lambda_s_A: float,
    center_distance_A: float,
    beta_eV_inv: float,
    k_eV_per_A2: float,
    *,
    nx: int,
    nrho: int,
) -> float:
    effective_radius = vdw_radius_A + lambda_s_A
    alpha = 0.5 * beta_eV_inv * k_eV_per_A2
    tail_extent = 9.0 / np.sqrt(alpha)
    half_distance = 0.5 * center_distance_A
    x = np.linspace(
        -half_distance - effective_radius - tail_extent,
        half_distance + effective_radius + tail_extent,
        nx,
    )
    rho = np.linspace(0.0, effective_radius + tail_extent, nrho)
    slice_integrals = np.empty_like(x)

    # Independent cylindrical-coordinate quadrature of the exact restraint:
    # exp[-beta*k/2*max(min_i(|r-c_i|-R)-lambda_s, 0)^2].
    for start in range(0, nx, 128):
        stop = min(start + 128, nx)
        x_block = x[start:stop, None]
        rho_block = rho[None, :]
        signed_distance = np.minimum(
            np.sqrt((x_block - half_distance) ** 2 + rho_block**2)
            - vdw_radius_A,
            np.sqrt((x_block + half_distance) ** 2 + rho_block**2)
            - vdw_radius_A,
        )
        delta = np.maximum(signed_distance - lambda_s_A, 0.0)
        integrand = (
            2.0
            * np.pi
            * rho_block
            * np.exp(-0.5 * beta_eV_inv * k_eV_per_A2 * delta**2)
        )
        slice_integrals[start:stop] = np.trapezoid(
            integrand,
            rho,
            axis=1,
        )

    return float(np.trapezoid(slice_integrals, x))


def test_single_sphere_veff_includes_harmonic_soft_tail_and_rejects_hard_volume():
    R = load_json(RADIUS_PATH)["radii"]["O"]
    protocol = _load_protocol()
    lambda_s = float(protocol["shell"]["lambda_s_candidates"]["candidate_offsets_A"][1]["offset_A"]) + 3.0
    effective_radius = R + lambda_s
    beta_eV_inv = 1.0 / (
        _kb_ev_per_k() * float(protocol["standard_state"]["temperature_k"])
    )
    k_eV_per_A2 = _wall_stiffness_kbt_eVA2(protocol)

    hard_volume = 4.0 / 3.0 * np.pi * effective_radius ** 3
    analytic_veff = _single_sphere_soft_veff_closed_form(
        effective_radius,
        beta_eV_inv,
        k_eV_per_A2,
    )

    alpha = 0.5 * beta_eV_inv * k_eV_per_A2
    tail_extent = 9.0 / np.sqrt(alpha)
    radial = np.linspace(0.0, effective_radius + tail_extent, 100001)
    delta = np.maximum(radial - effective_radius, 0.0)
    boltzmann = np.exp(-0.5 * beta_eV_inv * k_eV_per_A2 * delta ** 2)
    quadrature_veff = float(np.trapezoid(4.0 * np.pi * radial ** 2 * boltzmann, radial))

    rel_err = abs(quadrature_veff - analytic_veff) / analytic_veff
    assert rel_err <= 2.0e-6
    assert analytic_veff > hard_volume

    # Mutation guard: replacing the restraint integral by the geometric
    # indicator volume changes the standard-state correction materially.
    beta_kcal_inv = 1.0 / (
        float(protocol["thermodynamics"]["R_gas_kcal_per_mol_K"])
        * float(protocol["standard_state"]["temperature_k"])
    )
    hard_substitution_error_kcal = abs(
        -math.log(hard_volume / analytic_veff) / beta_kcal_inv
    )
    assert hard_substitution_error_kcal > 0.02


def test_two_sphere_hard_union_geometry_matches_independent_quadrature():
    R = load_json(RADIUS_PATH)["radii"]["O"]
    protocol = _load_protocol()
    lambda_s = float(protocol["shell"]["lambda_s_candidates"]["candidate_offsets_A"][1]["offset_A"]) + 3.0
    effective_radius = R + lambda_s
    analytic_single = 4.0 / 3.0 * np.pi * effective_radius ** 3
    d = effective_radius * 0.55
    analytic_union = 2.0 * analytic_single - _equal_sphere_overlap_volume(effective_radius, d)

    q1 = _union_volume_trapezoid(effective_radius, d, 18001)
    q2 = _union_volume_trapezoid(effective_radius, d, 4001)

    rel_err_q1 = abs(q1 - analytic_union) / analytic_union
    rel_err_q2 = abs(q2 - analytic_union) / analytic_union
    rel_diff = abs(q1 - q2) / analytic_union

    assert rel_err_q1 <= 1.5e-3
    assert rel_err_q2 <= 1.5e-3
    assert rel_diff <= 1.0e-3


def test_two_sphere_union_veff_integrates_the_actual_soft_minimum_restraint():
    vdw_radius = load_json(RADIUS_PATH)["radii"]["O"]
    protocol = _load_protocol()
    lambda_s = (
        float(
            protocol["shell"]["lambda_s_candidates"]["candidate_offsets_A"][
                1
            ]["offset_A"]
        )
        + 3.0
    )
    effective_radius = vdw_radius + lambda_s
    center_distance = 0.55 * effective_radius
    beta_eV_inv = 1.0 / (
        _kb_ev_per_k() * float(protocol["standard_state"]["temperature_k"])
    )
    k_eV_per_A2 = _wall_stiffness_kbt_eVA2(protocol)

    reference = _two_sphere_soft_veff_cylindrical(
        vdw_radius,
        lambda_s,
        center_distance,
        beta_eV_inv,
        k_eV_per_A2,
        nx=3201,
        nrho=2401,
    )
    independent_coarser = _two_sphere_soft_veff_cylindrical(
        vdw_radius,
        lambda_s,
        center_distance,
        beta_eV_inv,
        k_eV_per_A2,
        nx=1601,
        nrho=1201,
    )
    assert abs(reference - independent_coarser) / reference < 1.0e-3

    hard_single = 4.0 / 3.0 * np.pi * effective_radius**3
    hard_union = 2.0 * hard_single - _equal_sphere_overlap_volume(
        effective_radius,
        center_distance,
    )
    assert reference > hard_union

    beta_kcal_inv = 1.0 / (
        float(protocol["thermodynamics"]["R_gas_kcal_per_mol_K"])
        * float(protocol["standard_state"]["temperature_k"])
    )
    hard_substitution_error_kcal = abs(
        -math.log(hard_union / reference) / beta_kcal_inv
    )
    assert hard_substitution_error_kcal > 0.02

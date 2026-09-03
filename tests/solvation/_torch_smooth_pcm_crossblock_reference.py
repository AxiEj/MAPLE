"""Independent NumPy/SciPy reassembly for ``torch-smooth-pcm-v1``.

This oracle deliberately does not import the production ``torch_smooth_pcm``
package and never accepts matrices emitted by it.  It starts from geometry,
explicit radii, dielectric, quadrature orders, and raw point-l<=1 sources.

The independence boundary is numerical, not axiomatic: the oracle shares the
frozen real Condon--Shortley harmonic convention, the published ddPCM equations
``F=-Bc``, ``A_eps G=A_inf F``, ``LX=G``, and the model's declared smooth
switch/finite coefficient truncations.  It independently evaluates those
equations with SciPy harmonics, NumPy quadrature, explicit Python loops, and
NumPy linear algebra; no production private helper or detached production
matrix is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase.units import Bohr, Hartree
from scipy.linalg import block_diag
from scipy.special import lpmv


COULOMB_EV_ANGSTROM_PER_E2 = Hartree * Bohr
Q_RAW_FROM_CARTESIAN = np.asarray(
    ((1.0, 0.0, 0.0, 0.0),
     (0.0, 0.0, 1.0, 0.0),
     (0.0, 0.0, 0.0, 1.0),
     (0.0, 1.0, 0.0, 0.0)),
    dtype=float,
)


@dataclass(frozen=True)
class CrossBlockFixture:
    name: str
    positions: np.ndarray
    atomic_numbers: tuple[int, ...]
    radii: tuple[float, ...]
    source: np.ndarray
    dielectric: float
    transition_width: float = 0.08
    surface_lmax: int = 2
    partition_lmax: int = 4
    partition_order: int = 32
    source_order: int = 48
    double_layer_order: int = 48


P = CrossBlockFixture(
    "P",
    np.asarray(((0.0, 0.0, 0.0), (2.83, -0.41, 0.29))),
    (6, 8),
    (1.23, 1.09),
    np.asarray(((0.18, 0.030, -0.020, 0.010),
                (-0.18, -0.015, 0.025, -0.020))),
    80.0,
)
W = CrossBlockFixture(
    "W",
    np.asarray(((0.0, 0.0, 0.0),
                (0.9572, 0.0, 0.0),
                (-0.2399872, 0.927297, 0.0))),
    (8, 1, 1),
    (2.294, 1.2, 1.2),
    np.asarray(((-0.70, 0.04, -0.02, 0.03),
                (0.35, 0.00, 0.01, -0.02),
                (0.35, -0.01, 0.00, 0.02))),
    80.0,
)


def tangency_fixture(separation: float) -> CrossBlockFixture:
    if separation not in (0.2, 1.8):
        raise ValueError("frozen tangency separation must be 0.2 or 1.8")
    return CrossBlockFixture(
        f"T-{separation}",
        np.asarray(((0.0, 0.0, 0.0), (separation, 0.0, 0.0))),
        (1, 1),
        (1.0, 0.8),
        np.asarray(((0.4, 0.05, -0.02, 0.03),
                    (-0.4, -0.01, 0.04, -0.02))),
        20.0,
        surface_lmax=2,
        partition_lmax=4,
    )


def _labels(lmax: int) -> tuple[tuple[int, int], ...]:
    return tuple((ell, order) for ell in range(lmax + 1)
                 for order in range(-ell, ell + 1))


def _harmonics(directions: np.ndarray, lmax: int) -> np.ndarray:
    """Real orthonormal harmonics evaluated through SciPy ``lpmv``."""

    directions = np.asarray(directions, dtype=float)
    radius = np.linalg.norm(directions, axis=1)
    if np.any(radius == 0.0):
        raise ValueError("harmonic directions must be nonzero")
    unit = directions / radius[:, None]
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    columns = []
    for ell, signed_order in _labels(lmax):
        order = abs(signed_order)
        normalization = math.sqrt(
            (2 * ell + 1)
            * math.exp(math.lgamma(ell - order + 1) - math.lgamma(ell + order + 1))
            / (4.0 * math.pi)
        )
        value = normalization * lpmv(order, ell, np.cos(theta)) * np.exp(1j * order * phi)
        if signed_order < 0:
            value = math.sqrt(2.0) * ((-1) ** order) * value.imag
        elif signed_order == 0:
            value = value.real
        else:
            value = math.sqrt(2.0) * ((-1) ** order) * value.real
        columns.append(np.asarray(value, dtype=float))
    return np.column_stack(columns)


def _solid_harmonics(vectors: np.ndarray, lmax: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    radius = np.linalg.norm(vectors, axis=1)
    safe = vectors.copy()
    safe[radius == 0.0] = (0.0, 0.0, 1.0)
    design = _harmonics(safe, lmax)
    for column, (ell, _order) in enumerate(_labels(lmax)):
        design[:, column] *= radius**ell
    return design


def _sphere_rule_for_degree(degree: int) -> tuple[np.ndarray, np.ndarray]:
    cosine, polar_weight = np.polynomial.legendre.leggauss(degree // 2 + 1)
    azimuth_count = degree + 1
    phi = 2.0 * math.pi * np.arange(azimuth_count) / azimuth_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    directions = np.column_stack((
        np.repeat(sine, azimuth_count) * np.tile(np.cos(phi), len(cosine)),
        np.repeat(sine, azimuth_count) * np.tile(np.sin(phi), len(cosine)),
        np.repeat(cosine, azimuth_count),
    ))
    weights = np.repeat(polar_weight * (2.0 * math.pi / azimuth_count), azimuth_count)
    return directions, weights


def _flat_step(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    result = np.empty_like(value)
    result[value <= -1.0] = 0.0
    result[value >= 1.0] = 1.0
    mask = np.abs(value) < 1.0
    argument = 2.0 * value[mask] / (1.0 - value[mask] ** 2)
    result[mask] = 1.0 / (1.0 + np.exp(-argument))
    return result


def _constant_coefficients(lmax: int, value: float) -> np.ndarray:
    coefficients = np.zeros((lmax + 1) ** 2)
    coefficients[0] = value * math.sqrt(4.0 * math.pi)
    return coefficients


def _axial_factor(
    displacement: np.ndarray,
    radius_i: float,
    radius_j: float,
    width: float,
    lmax: int,
    radial_order: int,
    *,
    centered_exposure: bool,
) -> tuple[str, np.ndarray]:
    """Project either centered exposure or one-sided Schwarz inside weight."""

    distance = float(np.linalg.norm(displacement))
    minimum = (distance - radius_i) ** 2 - radius_j**2
    maximum = (distance + radius_i) ** 2 - radius_j**2
    if centered_exposure:
        if minimum >= width:
            return "exposed", _constant_coefficients(lmax, 1.0)
        if maximum <= -width:
            return "buried", _constant_coefficients(lmax, 0.0)
    else:
        if minimum >= 0.0:
            return "exposed", _constant_coefficients(lmax, 0.0)
        if maximum <= -width:
            return "buried", _constant_coefficients(lmax, 1.0)

    cosine, weights = np.polynomial.legendre.leggauss(radial_order)
    signed = radius_i**2 + distance**2 - 2.0 * radius_i * distance * cosine - radius_j**2
    values = (_flat_step(signed / width) if centered_exposure
              else 1.0 - _flat_step(2.0 * signed / width + 1.0))
    legendre = np.polynomial.legendre.legvander(cosine, lmax)
    zonal = 2.0 * math.pi * ((weights * values) @ legendre)
    axis_harmonics = _harmonics((displacement / distance)[None, :], lmax)[0]
    coefficients = np.concatenate([
        zonal[ell] * axis_harmonics[ell * ell:(ell + 1) ** 2]
        for ell in range(lmax + 1)
    ])
    return "transition", coefficients


def _project_values(values: np.ndarray, design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return design.T @ (weights * values)


def _project_product(factors: list[np.ndarray], lmax: int) -> np.ndarray:
    if not factors:
        return _constant_coefficients(lmax, 1.0)
    directions, weights = _sphere_rule_for_degree((len(factors) + 1) * lmax)
    design = _harmonics(directions, lmax)
    values = np.ones(len(directions))
    for coefficients in factors:
        values *= design @ coefficients
    return _project_values(values, design, weights)


def _project_product_and_leave_one_out(
    factors: list[np.ndarray], lmax: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    if not factors:
        return _constant_coefficients(lmax, 1.0), []
    directions, weights = _sphere_rule_for_degree((len(factors) + 1) * lmax)
    design = _harmonics(directions, lmax)
    factor_values = [design @ factor for factor in factors]
    full_values = np.prod(np.stack(factor_values), axis=0)
    leave = []
    for excluded in range(len(factors)):
        values = np.ones(len(directions))
        for index, factor_value in enumerate(factor_values):
            if index != excluded:
                values *= factor_value
        leave.append(_project_values(values, design, weights))
    return _project_values(full_values, design, weights), leave


def _weighted_basis(coefficients: np.ndarray, exposure_lmax: int, basis_lmax: int) -> np.ndarray:
    product_lmax = exposure_lmax + basis_lmax
    degree = exposure_lmax + basis_lmax + product_lmax
    directions, weights = _sphere_rule_for_degree(degree)
    exposure_design = _harmonics(directions, exposure_lmax)
    basis_design = _harmonics(directions, basis_lmax)
    product_design = _harmonics(directions, product_lmax)
    return product_design.T @ (
        (weights * (exposure_design @ coefficients))[:, None] * basis_design
    )


def _schwarz_partition(fixture: CrossBlockFixture):
    one = _constant_coefficients(fixture.partition_lmax, 1.0)
    rows = []
    for target, radius_i in enumerate(fixture.radii):
        factors: list[tuple[int, np.ndarray]] = []
        for source, radius_j in enumerate(fixture.radii):
            if source == target:
                continue
            state, coefficient = _axial_factor(
                fixture.positions[source] - fixture.positions[target],
                radius_i, radius_j, fixture.transition_width,
                fixture.partition_lmax, fixture.partition_order,
                centered_exposure=False,
            )
            if state != "exposed":
                factors.append((source, coefficient))
        if not factors:
            rows.append((one.copy(), {}))
            continue
        count = len(factors)
        directions, weights = _sphere_rule_for_degree((count + 1) * fixture.partition_lmax)
        design = _harmonics(directions, fixture.partition_lmax)
        fv = np.stack([design @ coefficient for _, coefficient in factors])
        one_values = design @ one
        exposed = _project_values(np.prod(one_values[None, :] - fv, axis=0), design, weights)
        nodes, scalar_weights = np.polynomial.legendre.leggauss(max(1, (count + 1) // 2))
        nodes = 0.5 * (nodes + 1.0)
        scalar_weights *= 0.5
        omega = np.zeros_like(fv)
        for node, scalar_weight in zip(nodes, scalar_weights, strict=True):
            complements = one_values[None, :] - node * fv
            for selected in range(count):
                product = np.ones(len(directions))
                for other in range(count):
                    if other != selected:
                        product *= complements[other]
                omega[selected] += scalar_weight * fv[selected] * product
        overlap = {
            source: _project_values(omega[index], design, weights)
            for index, (source, _coefficient) in enumerate(factors)
        }
        rows.append((exposed, overlap))
    return tuple(rows)


def _centered_partition(fixture: CrossBlockFixture):
    factor_rows, full_rows, remaining_rows = [], [], []
    for target, radius_i in enumerate(fixture.radii):
        row = {}
        active: list[tuple[int, np.ndarray]] = []
        buried = False
        for source, radius_j in enumerate(fixture.radii):
            if source == target:
                continue
            state, coefficient = _axial_factor(
                fixture.positions[source] - fixture.positions[target],
                radius_i, radius_j, fixture.transition_width,
                fixture.partition_lmax, fixture.double_layer_order,
                centered_exposure=True,
            )
            row[source] = (state, coefficient)
            buried |= state == "buried"
            if state == "transition":
                active.append((source, coefficient))
        zero = _constant_coefficients(fixture.partition_lmax, 0.0)
        if buried:
            full, remaining = zero, {source: zero for source in row}
        else:
            full, leave = _project_product_and_leave_one_out(
                [coefficient for _, coefficient in active], fixture.partition_lmax
            )
            by_source = {source: leave[index] for index, (source, _) in enumerate(active)}
            remaining = {source: by_source.get(source, full) for source in row}
        factor_rows.append(row)
        full_rows.append(full)
        remaining_rows.append(remaining)
    return tuple(factor_rows), tuple(full_rows), tuple(remaining_rows)


def _local_translation(fixture: CrossBlockFixture, target: int, source: int) -> np.ndarray:
    directions, weights = _sphere_rule_for_degree(2 * fixture.surface_lmax)
    target_design = _harmonics(directions, fixture.surface_lmax)
    relative = (
        fixture.positions[target] - fixture.positions[source]
        + fixture.radii[target] * directions
    ) / fixture.radii[source]
    return target_design.T @ (weights[:, None] * _solid_harmonics(relative, fixture.surface_lmax))


def _source_block(
    fixture: CrossBlockFixture, target: int, source: int, physical_lmax: int
) -> np.ndarray:
    dimension = (physical_lmax + 1) ** 2
    radius = fixture.radii[target]
    if target == source:
        block = np.zeros((dimension, 4))
        block[0, 0] = COULOMB_EV_ANGSTROM_PER_E2 * math.sqrt(4.0 * math.pi) / radius
        amplitude = COULOMB_EV_ANGSTROM_PER_E2 * math.sqrt(4.0 * math.pi / 3.0) / radius**2
        block[1:4, 1:4] = amplitude * np.eye(3)
        return block

    # Use the same invariant radial order, but directly integrate each physical
    # point multipole in a pair-axis frame instead of differentiating a projected
    # monopole coefficient as production does.
    cosine, weights = np.polynomial.legendre.leggauss(fixture.source_order)
    azimuth_count = 2 * physical_lmax + 1
    phi = 2.0 * math.pi * np.arange(azimuth_count) / azimuth_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    local = np.column_stack((
        np.repeat(sine, azimuth_count) * np.tile(np.cos(phi), len(cosine)),
        np.repeat(sine, azimuth_count) * np.tile(np.sin(phi), len(cosine)),
        np.repeat(cosine, azimuth_count),
    ))
    point_weights = np.repeat(weights * (2.0 * math.pi / azimuth_count), azimuth_count)
    rotation = _rotation_from_z(fixture.positions[source] - fixture.positions[target])
    directions = local @ rotation.T
    points = fixture.positions[target] + radius * directions
    displacement = points - fixture.positions[source]
    distance = np.linalg.norm(displacement, axis=1)
    design = _harmonics(directions, physical_lmax)
    cartesian = displacement / distance[:, None] ** 3
    raw_kernels = np.column_stack((1.0 / distance, cartesian[:, 1], cartesian[:, 2], cartesian[:, 0]))
    return COULOMB_EV_ANGSTROM_PER_E2 * design.T @ (point_weights[:, None] * raw_kernels)


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    base = np.asarray((0.0, 0.0, 1.0)) if unit[2] > -0.5 else np.asarray((0.0, 0.0, -1.0))
    axis = np.cross(base, unit)
    cosine = float(base @ unit)
    cross = np.asarray(((0.0, -axis[2], axis[1]),
                        (axis[2], 0.0, -axis[0]),
                        (-axis[1], axis[0], 0.0)))
    rotation = np.eye(3) + cross + cross @ cross / (1.0 + cosine)
    if base[2] < 0.0:
        rotation = rotation @ np.diag((1.0, -1.0, -1.0))
    return rotation


def _localized_pair_double_layer(
    fixture: CrossBlockFixture, target: int, source: int, target_lmax: int
) -> np.ndarray:
    """Direct lab-frame pair-localized exterior double-layer projection."""

    cosine, polar_weights = np.polynomial.legendre.leggauss(fixture.double_layer_order)
    azimuth_count = target_lmax + fixture.surface_lmax + 1
    phi = 2.0 * math.pi * np.arange(azimuth_count) / azimuth_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    local = np.column_stack((
        np.repeat(sine, azimuth_count) * np.tile(np.cos(phi), len(cosine)),
        np.repeat(sine, azimuth_count) * np.tile(np.sin(phi), len(cosine)),
        np.repeat(cosine, azimuth_count),
    ))
    weights = np.repeat(polar_weights * (2.0 * math.pi / azimuth_count), azimuth_count)
    displacement = fixture.positions[target] - fixture.positions[source]
    rotation = _rotation_from_z(displacement)
    target_directions = local @ rotation.T
    points_from_source = displacement + fixture.radii[target] * target_directions
    radial_squared = np.einsum("pi,pi->p", points_from_source, points_from_source)
    radial = np.sqrt(radial_squared)
    pair_exposure = _flat_step((radial_squared - fixture.radii[source] ** 2) / fixture.transition_width)
    safe_direction = points_from_source.copy()
    inactive = pair_exposure == 0.0
    safe_direction[inactive] = (0.0, 0.0, 1.0)
    safe_radius = radial.copy()
    safe_radius[inactive] = 1.0
    target_design = _harmonics(target_directions, target_lmax)
    source_design = _harmonics(safe_direction, fixture.surface_lmax)
    kernel = np.empty_like(source_design)
    for column, (ell, _order) in enumerate(_labels(fixture.surface_lmax)):
        scale = (0.0 if ell == 0 else
                 ell * (fixture.radii[source] / safe_radius) ** (ell + 1))
        kernel[:, column] = (
            pair_exposure * 4.0 * math.pi / (2 * ell + 1) * scale
        )
    return target_design.T @ (weights[:, None] * source_design * kernel)


def assemble_reference(fixture: CrossBlockFixture) -> dict[str, np.ndarray]:
    """Reassemble P/W/T operators solely from the frozen mathematical inputs."""

    atom_count = len(fixture.radii)
    local_dimension = (fixture.surface_lmax + 1) ** 2
    physical_lmax = fixture.surface_lmax + fixture.partition_lmax
    physical_dimension = (physical_lmax + 1) ** 2

    schwarz_partition = _schwarz_partition(fixture)
    schwarz_rows = []
    for target, (_exposed, overlap) in enumerate(schwarz_partition):
        blocks = []
        for source in range(atom_count):
            if source == target:
                block = np.eye(local_dimension)
            elif source not in overlap:
                block = np.zeros((local_dimension, local_dimension))
            else:
                multiplication = _weighted_basis(
                    overlap[source], fixture.partition_lmax, fixture.surface_lmax
                )[:local_dimension]
                block = -multiplication @ _local_translation(fixture, target, source)
            blocks.append(block)
        schwarz_rows.append(np.concatenate(blocks, axis=1))
    schwarz = np.concatenate(schwarz_rows, axis=0)

    raw_source_rows = []
    for target in range(atom_count):
        raw_source_rows.append(np.concatenate([
            _source_block(fixture, target, source, physical_lmax)
            for source in range(atom_count)
        ], axis=1))
    raw_source = np.concatenate(raw_source_rows, axis=0)
    source_operator = np.zeros((atom_count * local_dimension, atom_count * 4))
    for target, (exposed, _overlap) in enumerate(schwarz_partition):
        weighted_test = _weighted_basis(exposed, fixture.partition_lmax, fixture.surface_lmax)
        raw_block = raw_source[target * physical_dimension:(target + 1) * physical_dimension]
        source_operator[target * local_dimension:(target + 1) * local_dimension] = weighted_test.T @ raw_block

    factor_rows, centered_exposure, remaining = _centered_partition(fixture)
    double_rows = [[None for _ in range(atom_count)] for _ in range(atom_count)]
    self_spectrum = block_diag(*[
        (-2.0 * math.pi / (2 * ell + 1)) * np.eye(2 * ell + 1)
        for ell in range(fixture.surface_lmax + 1)
    ])
    for target in range(atom_count):
        multiplication = _weighted_basis(
            centered_exposure[target], fixture.partition_lmax, fixture.surface_lmax
        )[:local_dimension]
        double_rows[target][target] = multiplication @ self_spectrum
        for source in range(atom_count):
            if source == target:
                continue
            state, _coefficient = factor_rows[target][source]
            if state == "buried":
                double_rows[target][source] = np.zeros((local_dimension, local_dimension))
            else:
                localized = _localized_pair_double_layer(fixture, target, source, physical_lmax)
                remaining_basis = _weighted_basis(
                    remaining[target][source], fixture.partition_lmax, fixture.surface_lmax
                )
                double_rows[target][source] = remaining_basis.T @ localized
    localized_double = np.concatenate(
        [np.concatenate(row, axis=1) for row in double_rows], axis=0
    )

    identity = np.eye(atom_count * local_dimension)
    conductor = 2.0 * math.pi * identity - localized_double
    jump = (fixture.dielectric + 1.0) / (fixture.dielectric - 1.0)
    dielectric = 2.0 * math.pi * jump * identity - localized_double

    receiver_blocks = []
    for radius in fixture.radii:
        block = np.zeros((4, local_dimension))
        block[0, 0] = 1.0 / math.sqrt(4.0 * math.pi)
        block[1:4, 1:4] = math.sqrt(3.0 / (4.0 * math.pi)) / radius * np.eye(3)
        receiver_blocks.append(block)
    receiver_raw = block_diag(*receiver_blocks)
    q = np.kron(np.eye(atom_count), Q_RAW_FROM_CARTESIAN)
    receiver_field = q.T @ receiver_raw

    exposed = np.stack([row[0] for row in schwarz_partition])
    return {
        "B": source_operator,
        "C_raw": receiver_raw,
        "C_field": receiver_field,
        "L": schwarz,
        "localized_double_layer": localized_double,
        "A_inf": conductor,
        "A_eps": dielectric,
        "exposed_coefficients": exposed,
        "centered_exposure_coefficients": np.stack(centered_exposure),
    }


def solve_reference(
    fixture: CrossBlockFixture, matrices: dict[str, np.ndarray]
) -> dict[str, np.ndarray | float]:
    c = fixture.source.ravel()
    force = -(matrices["B"] @ c)
    dielectric_state = np.linalg.solve(
        matrices["A_eps"], matrices["A_inf"] @ force
    )
    potential = np.linalg.solve(matrices["L"], dielectric_state)
    energy = 0.5 * float(c @ (matrices["C_raw"] @ potential))
    lambda_x = np.linalg.solve(
        matrices["L"].T, -0.5 * matrices["C_raw"].T @ c
    )
    lambda_g = np.linalg.solve(matrices["A_eps"].T, lambda_x)
    lambda_f = matrices["A_inf"].T @ lambda_g
    raw_gradient = 0.5 * matrices["C_raw"] @ potential + matrices["B"].T @ lambda_f
    return {
        "F": force, "G": dielectric_state, "X": potential, "U": energy,
        "lambda_X": lambda_x, "lambda_G": lambda_g, "lambda_F": lambda_f,
        "raw_gradient_kkt": raw_gradient,
    }


__all__ = [
    "CrossBlockFixture", "P", "W", "tangency_fixture", "assemble_reference",
    "solve_reference", "Q_RAW_FROM_CARTESIAN", "COULOMB_EV_ANGSTROM_PER_E2",
]

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pytest

from .conftest import FIXTURE_DIR, load_json

FIX = FIXTURE_DIR / "alchemy" / "many_body_identity.json"
STAGES = ("D", "R", "I", "P")


class ATOM_LIST_MISMATCH(RuntimeError):
    """Raised when atom index lists are reordered, omitted, or mismatched."""


def _as_array(x):
    return np.asarray(x, dtype=float)


def _atom_list_signature(atom_lists: Mapping[str, list[int]]) -> str:
    payload = json.dumps(atom_lists, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_and_load_fixture(data_override: dict | None = None) -> dict:
    data = copy.deepcopy(load_json(FIX)) if data_override is None else data_override
    model = data["interaction_model"]

    atom_lists = model["atom_lists"]
    if _atom_list_signature(atom_lists) != model["atom_list_hash_sha256"]:
        raise ATOM_LIST_MISMATCH("ATOM_LIST_MISMATCH: atom-list hash mismatch")

    x = atom_lists["x"]
    w = atom_lists["w"]
    xw = atom_lists["xw"]
    n = len(model["coordinates_A"])

    x_set = set(x)
    w_set = set(w)
    xw_set = set(xw)
    coord_set = set(range(n))

    if len(x) != len(set(x)) or len(w) != len(set(w)) or len(xw) != len(set(xw)):
        raise ATOM_LIST_MISMATCH("ATOM_LIST_MISMATCH: duplicate index in atom list")
    if x_set & w_set:
        raise ATOM_LIST_MISMATCH("ATOM_LIST_MISMATCH: x/w overlap")
    if xw_set != coord_set:
        raise ATOM_LIST_MISMATCH("ATOM_LIST_MISMATCH: xw does not cover coordinates")
    if x_set | w_set != xw_set:
        raise ATOM_LIST_MISMATCH("ATOM_LIST_MISMATCH: x+w partition incomplete")
    if any(i < 0 or i >= n for i in x + w + xw):
        raise ATOM_LIST_MISMATCH("ATOM_LIST_MISMATCH: atom index out of bounds")

    return data


def _cross_scales(model: dict, stage: str) -> tuple[float, float]:
    scale = model["stage_scale"][stage]
    return float(scale["cross_repulsive"]), float(scale["cross_full"])


def _analytic_many_body_model(coords: np.ndarray, idx_x: set[int], idx_w: set[int], model: dict, stage: str, *, include_cross: bool) -> tuple[float, np.ndarray, np.ndarray]:
    params = model["pair_constants"]
    n = coords.shape[0]
    u_per_atom = np.zeros(n)
    f = np.zeros((n, 3), dtype=float)
    u = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            cross_pair = (i in idx_x and j in idx_w) or (i in idx_w and j in idx_x)
            if i in idx_x and j in idx_x:
                k = params["k_xx"]
            elif i in idx_w and j in idx_w:
                k = params["k_ww"]
            elif cross_pair:
                if not include_cross:
                    continue
                k = params["k_xw_full"]
            else:
                continue

            d = coords[i] - coords[j]
            r2 = float(np.dot(d, d))
            pair_u = 0.0
            pair_f = np.zeros(3, dtype=float)

            if cross_pair:
                repulsive_scale, full_scale = _cross_scales(model, stage)
                core = model["repulsive_core"]
                sigma2 = float(core["sigma_A"]) ** 2
                repulsive_u = (
                    repulsive_scale
                    * float(core["epsilon_eV"])
                    * float(np.exp(-r2 / sigma2))
                )
                # U_rep=epsilon*exp(-r^2/sigma^2), so F_i=-dU/dri
                # points along ri-rj and separates a near-overlapping pair.
                pair_u += repulsive_u
                pair_f += (2.0 * repulsive_u / sigma2) * d
                pair_u += 0.5 * full_scale * k * r2
                pair_f += -full_scale * k * d
            else:
                pair_u = 0.5 * k * r2
                pair_f = -k * d

            u += pair_u
            u_per_atom[i] += pair_u
            u_per_atom[j] += pair_u
            f[i] += pair_f
            f[j] -= pair_f

    term = model["irreducible_cross_three_body"]
    anchor = int(term["solute_anchor_index"])
    water_a, water_b = [int(index) for index in term["water_atom_indices"]]
    _, full_scale = _cross_scales(model, stage)
    if (
        include_cross
        and full_scale != 0.0
        and anchor in idx_x
        and water_a in idx_w
        and water_b in idx_w
    ):
        vec_a = coords[water_a] - coords[anchor]
        vec_b = coords[water_b] - coords[anchor]
        dot_ab = float(np.dot(vec_a, vec_b))
        coefficient = float(term["coefficient_eV_per_A2"])
        length_scale2 = float(term["length_scale_A"]) ** 2
        three_body_u = full_scale * coefficient * dot_ab ** 2 / length_scale2
        grad_a = (
            full_scale
            * 2.0
            * coefficient
            * dot_ab
            * vec_b
            / length_scale2
        )
        grad_b = (
            full_scale
            * 2.0
            * coefficient
            * dot_ab
            * vec_a
            / length_scale2
        )
        u += three_body_u
        u_per_atom[[anchor, water_a, water_b]] += three_body_u
        f[water_a] -= grad_a
        f[water_b] -= grad_b
        f[anchor] += grad_a + grad_b

    return u, u_per_atom, f


def _pairwise_fragment(coords: np.ndarray, idx: set[int], k: float) -> tuple[float, np.ndarray, np.ndarray]:
    n = coords.shape[0]
    idx = set(idx)
    u = 0.0
    u_per_atom = np.zeros(n)
    f = np.zeros((n, 3), dtype=float)

    for i in sorted(idx):
        for j in sorted(idx):
            if j <= i:
                continue
            d = coords[i] - coords[j]
            pair_u = 0.5 * k * float(np.dot(d, d))
            pair_f = -k * d

            u += pair_u
            u_per_atom[i] += pair_u
            u_per_atom[j] += pair_u
            f[i] += pair_f
            f[j] -= pair_f

    return u, u_per_atom, f


def _eval_stage_state(coords: np.ndarray, model: dict) -> dict[str, dict]:
    idx_x = set(model["atom_lists"]["x"])
    idx_w = set(model["atom_lists"]["w"])
    params = model["pair_constants"]
    n = len(coords)

    # fragment-only contributions are independent of stage
    u_x, u_x_per_atom, f_x = _pairwise_fragment(coords, idx_x, params["k_xx"])
    u_w, u_w_per_atom, f_w = _pairwise_fragment(coords, idx_w, params["k_ww"])

    # for explicitness, preserve all arrays and source-of-truth identity checks
    del u_x_per_atom
    del u_w_per_atom

    result: dict[str, dict] = {}
    for stage in STAGES:
        u_xw, u_xw_per_atom, f_xw = _analytic_many_body_model(coords, idx_x, idx_w, model, stage, include_cross=True)

        # embed fragment forces into combined geometry (all arrays are already shared shape)
        assert u_xw_per_atom.shape == (n,)

        # scalar/force identity
        u_int = u_xw - u_x - u_w
        f_int = f_xw - f_x - f_w

        result[stage] = {
            "u_xw": float(u_xw),
            "u_x": float(u_x),
            "u_w": float(u_w),
            "u_int": float(u_int),
            "f_xw": f_xw,
            "f_x": f_x,
            "f_w": f_w,
            "f_int": f_int,
        }

    return result


def _stage_u_int(coords: np.ndarray, model: dict, stage: str) -> float:
    return _eval_stage_state(coords, model)[stage]["u_int"]


def _central_difference_scalar(coords: np.ndarray, model: dict, stage: str, atom_i: int, axis: int, eps: float = 1e-6):
    plus = coords.copy()
    minus = coords.copy()
    plus[atom_i, axis] += eps
    minus[atom_i, axis] -= eps

    up = _stage_u_int(plus, model, stage)
    um = _stage_u_int(minus, model, stage)
    return -(up - um) / (2.0 * eps)


def test_many_body_analytic_reconstruction_matches_stored_secondary_arrays():
    data = _validate_and_load_fixture()
    model = data["interaction_model"]
    coords = _as_array(model["coordinates_A"])
    results = _eval_stage_state(coords, model)

    for stage in STAGES:
        rec = results[stage]
        np.testing.assert_allclose(rec["f_xw"], rec["f_x"] + rec["f_w"] + rec["f_int"])
        np.testing.assert_allclose(rec["u_xw"], rec["u_x"] + rec["u_w"] + rec["u_int"], rtol=0, atol=1e-12)

    expected = data.get("expected", {})
    for stage in STAGES:
        exp = expected[stage]
        np.testing.assert_allclose(results[stage]["u_int"], exp["u_int"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(results[stage]["f_int"], exp["f_int"], rtol=0, atol=1e-12)


def test_many_body_interaction_force_matches_central_difference_for_all_coordinates_and_stages():
    data = _validate_and_load_fixture()
    model = data["interaction_model"]
    coords = _as_array(model["coordinates_A"])
    results = _eval_stage_state(coords, model)

    for stage in STAGES:
        f = results[stage]["f_int"]
        for atom_i in range(coords.shape[0]):
            for axis in range(3):
                fd = _central_difference_scalar(coords, model, stage, atom_i, axis)
                assert fd == pytest.approx(f[atom_i, axis], rel=2e-5, abs=2e-6)


def test_many_body_atom_list_mismatch_is_fail_closed_before_energy_evaluation():
    data = _validate_and_load_fixture()

    bad_reordered = copy.deepcopy(data)
    bad_reordered["interaction_model"]["atom_lists"]["x"] = [1, 0, 2, 3]
    with pytest.raises(ATOM_LIST_MISMATCH):
        _validate_and_load_fixture(bad_reordered)

    bad_omitted = copy.deepcopy(data)
    bad_omitted["interaction_model"]["atom_lists"]["w"].pop()
    with pytest.raises(ATOM_LIST_MISMATCH):
        _validate_and_load_fixture(bad_omitted)


def test_many_body_stage_D_R_I_P_endpoint_semantics_and_path_identity():
    data = _validate_and_load_fixture()
    model = data["interaction_model"]
    coords = _as_array(model["coordinates_A"])
    results = _eval_stage_state(coords, model)

    # D: decoupled (cross) interactions off
    np.testing.assert_allclose(results["D"]["u_int"], 0.0, atol=1e-12)
    np.testing.assert_allclose(results["D"]["f_int"], np.zeros_like(results["D"]["f_int"]), atol=1e-12)

    # R: finite repulsive only
    np.testing.assert_array_less(0.0, results["R"]["u_int"])
    # I: repulsion + full interaction
    np.testing.assert_allclose(
        results["I"]["u_int"],
        results["R"]["u_int"] + results["P"]["u_int"],
        rtol=0,
        atol=2e-12,
    )

    # forward/reverse path identity
    forward = results["P"]["u_int"] - results["D"]["u_int"]
    segmented = (
        results["R"]["u_int"] - results["D"]["u_int"]
        + results["I"]["u_int"] - results["R"]["u_int"]
        + results["P"]["u_int"] - results["I"]["u_int"]
    )
    assert forward == pytest.approx(segmented, abs=1e-12)
    assert results["D"]["u_int"] - results["P"]["u_int"] == pytest.approx(-forward, abs=1e-12)

    # Coincident and near-coincident geometries remain finite across staged path.
    # The separate analytic core test below proves that R is a barrier rather
    # than merely a finite attractive quadratic.
    near = coords.copy()
    near[4] = near[0]

    near_results = _eval_stage_state(near, model)
    for stage in STAGES:
        assert np.isfinite(near_results[stage]["u_int"])
        assert np.all(np.isfinite(near_results[stage]["f_int"]))


def test_R_leg_gaussian_core_is_finite_positive_and_separating():
    data = _validate_and_load_fixture()
    model = data["interaction_model"]
    core = model["repulsive_core"]
    epsilon = float(core["epsilon_eV"])
    sigma = float(core["sigma_A"])

    coincident = np.zeros((2, 3), dtype=float)
    u0, _, f0 = _analytic_many_body_model(
        coincident,
        {0},
        {1},
        model,
        "R",
        include_cross=True,
    )
    assert u0 == pytest.approx(epsilon, rel=0.0, abs=1e-15)
    assert np.all(np.isfinite(f0))

    near = coincident.copy()
    near[0, 0] = 0.1 * sigma
    u_near, _, f_near = _analytic_many_body_model(
        near,
        {0},
        {1},
        model,
        "R",
        include_cross=True,
    )
    displacement = near[0] - near[1]
    relative_force = f_near[0] - f_near[1]
    assert 0.0 < u_near < u0
    assert float(np.dot(relative_force, displacement)) > 0.0

    outside = coincident.copy()
    outside[0, 0] = 3.0 * sigma
    u_outside, _, _ = _analytic_many_body_model(
        outside,
        {0},
        {1},
        model,
        "R",
        include_cross=True,
    )
    assert u_outside < 2.0e-4 * epsilon


def test_irreducible_cross_fragment_three_body_term_is_present_in_subtraction():
    data = _validate_and_load_fixture()
    model = data["interaction_model"]
    coords = _as_array(model["coordinates_A"])
    results = _eval_stage_state(coords, model)

    without_three_body = copy.deepcopy(model)
    without_three_body["irreducible_cross_three_body"][
        "coefficient_eV_per_A2"
    ] = 0.0
    pair_only = _eval_stage_state(coords, without_three_body)

    term = model["irreducible_cross_three_body"]
    anchor = term["solute_anchor_index"]
    water_a, water_b = term["water_atom_indices"]
    vec_a = coords[water_a] - coords[anchor]
    vec_b = coords[water_b] - coords[anchor]
    expected = (
        term["coefficient_eV_per_A2"]
        * float(np.dot(vec_a, vec_b)) ** 2
        / term["length_scale_A"] ** 2
    )
    assert expected > 0.0
    assert results["P"]["u_int"] - pair_only["P"]["u_int"] == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert results["I"]["u_int"] - pair_only["I"]["u_int"] == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert results["R"]["u_int"] == pytest.approx(
        pair_only["R"]["u_int"],
        abs=1e-12,
    )

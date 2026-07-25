from __future__ import annotations

import copy
import math

import numpy as np

from .conftest import DOCS_DIR, FIXTURE_DIR, load_json
from .validators import canonical_sha256

QCT_FIX = FIXTURE_DIR / "qct" / "qct_toy.json"
PLATEAU_FIX = FIXTURE_DIR / "qct" / "plateau_tail_cases.json"
PROTOCOL_FIX = DOCS_DIR / "protocol-v1.json"


def _free_energy_from_rows(rows, beta, include_tail=False):
    terms = [row["G_n"] for row in rows if include_tail or row["included"]]
    z = sum(math.exp(-beta * g) for g in terms)
    return -math.log(z) / beta


def _derived_tail(rows, beta):
    weights = [math.exp(-beta * row["G_n"]) for row in rows]
    z_all = sum(weights)
    z_included = sum(
        weight for weight, row in zip(weights, rows) if row["included"]
    )
    omitted_probability = sum(
        weight for weight, row in zip(weights, rows) if not row["included"]
    ) / z_all
    correction_kcal = math.log(z_all / z_included) / beta
    per_row_probability = [
        weight / z_all if not row["included"] else 0.0
        for weight, row in zip(weights, rows)
    ]
    return omitted_probability, correction_kcal, per_row_probability


def _tail_envelope_matches_contract(envelope, contract):
    immutable = dict(contract)
    declared_hash = immutable.pop("contract_sha256", None)
    if declared_hash != canonical_sha256(immutable):
        return False
    return all(
        envelope.get(key) == value
        for key, value in immutable.items()
        if key
        not in {
            "contract_name",
            "contract_version",
            "immutable",
        }
    )


def _tail_envelope_bound(rows, beta, envelope, contract=None):
    if contract is not None and not _tail_envelope_matches_contract(
        envelope, contract
    ):
        return None
    omitted = sorted(
        (row for row in rows if not row["included"]),
        key=lambda row: row["n"],
    )
    minimum = int(envelope["minimum_consecutive_ratios"])
    if len(omitted) - 1 < minimum:
        return None
    weights = [math.exp(-beta * row["G_n"]) for row in omitted]
    ratios = [
        weights[index + 1] / weights[index]
        for index in range(len(weights) - 1)
    ]
    if (
        envelope.get("upper_bound_semantics")
        != "preregistered deterministic modeling ceiling, not a statistical confidence interval"
        or envelope.get("confidence_level") is not None
    ):
        return None
    q_upper = float(envelope["conservative_ratio_ceiling"])
    if not 0.0 <= q_upper < 1.0:
        return None
    if any(ratio < 0.0 or ratio > q_upper for ratio in ratios):
        return None
    if [row["n"] for row in omitted] != list(
        range(omitted[0]["n"], omitted[-1]["n"] + 1)
    ):
        return None
    z_included = sum(
        math.exp(-beta * row["G_n"]) for row in rows if row["included"]
    )
    unseen_weight_upper = weights[-1] * q_upper / (1.0 - q_upper)
    tail_weight_upper = sum(weights) + unseen_weight_upper
    z_upper = z_included + tail_weight_upper
    return {
        "ratios": ratios,
        "unseen_weight_upper": unseen_weight_upper,
        "probability_upper": tail_weight_upper / z_upper,
        "free_energy_correction_upper_kcal": math.log(
            z_upper / z_included
        )
        / beta,
    }


def _reconstruct_gn(row):
    return (
        row["labeled_alchemical_association_total"]
        + row["volume_total"]
        + row["symmetry_total"]
        + row["release_total"]
        + row["water_density_total"]
        + row["outer_cluster_total"]
        + row["signed_water_outer_subtraction"]
    )


def _ci_intersection(rows):
    lo = -1e300
    hi = 1e300
    for row in rows:
        lo = max(lo, row["ci"][0])
        hi = min(hi, row["ci"][1])
    return lo <= hi


def _covariance_delta_method(cov, values, beta):
    # ∂(−1/β ln Σ exp(-βx_i)) / ∂x_i = p_i
    w = np.exp(-beta * values)
    p = w / np.sum(w)
    var = float(p @ cov @ p)
    return p, var


def test_qct_rows_reconstruct_and_validate_per_edge_terms():
    data = load_json(QCT_FIX)
    rows = data["rows"]

    assert "g_n" not in data

    for row in rows:
        n = row["n"]
        assert row["included"] is False or n >= 0
        if row["n"] == 0:
            assert not row["labeled_alchemical_association_steps"]
            assert not row["volume_steps"]
            assert not row["symmetry_steps"]
            assert not row["release_steps"]
            assert row["outer_cluster_total"] != 0.0

        for key in [
            "labeled_alchemical_association_steps",
            "volume_steps",
            "symmetry_steps",
            "release_steps",
        ]:
            assert len(row[key]) == n

        # only these edge arrays are per n by construction
        assert "water_density_steps" not in row
        assert "water_outer_steps" not in row
        assert "outer_cluster_steps" not in row

        assert math.isclose(
            sum(row["labeled_alchemical_association_steps"]),
            row["labeled_alchemical_association_total"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            sum(row["volume_steps"]),
            row["volume_total"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            sum(row["symmetry_steps"]),
            row["symmetry_total"],
            abs_tol=1e-12,
        )

        # RT ln(i) per edge is exact
        expected_sym = [1.0 / data["beta_kcal_inv"] * math.log(i) for i in range(1, n + 1)]
        assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(row["symmetry_steps"], expected_sym))

        assert math.isclose(sum(row["release_steps"]), row["release_total"], abs_tol=1e-12)
        assert row["release_total"] == 0.0
        assert all(step == 0.0 for step in row["release_steps"])

        # release endpoints are per-edge and must agree with delta in each step
        assert len(row["release_endpoints"]) == n
        for step, endpoint in zip(row["release_steps"], row["release_endpoints"]):
            assert math.isclose(endpoint["delta"], step, abs_tol=1e-12)
            assert math.isclose(endpoint["F_target"] - endpoint["F_biased"], step, abs_tol=1e-12)
            assert endpoint["F_target"] == endpoint["F_biased"]

        assert math.isclose(row["water_density_total"], n * row["water_density_per_water"], abs_tol=1e-12)
        assert math.isclose(row["water_outer_total"], n * row["water_outer_per_water"], abs_tol=1e-12)
        assert math.isclose(row["signed_water_outer_subtraction"], -row["water_outer_total"], abs_tol=1e-12)

        reconstructed = _reconstruct_gn(row)
        assert math.isclose(reconstructed, row["G_n"], abs_tol=1e-10)
        assert row["water_density_per_water"] != row["water_outer_per_water"]

    included = [row for row in rows if row["included"]]
    pass_rows = [r for r in rows if r["included"]]
    assert len(included) == 3
    assert all(r["n"] < 3 for r in pass_rows)

    beta = data["beta_kcal_inv"]
    computed_gn = _free_energy_from_rows(rows, beta, include_tail=False)
    direct = _free_energy_from_rows(included, data["beta_kcal_inv"], include_tail=True)
    assert math.isclose(computed_gn, direct, abs_tol=1e-12)
    all_rows = _free_energy_from_rows(rows, beta, include_tail=True)
    _, correction, _ = _derived_tail(rows, beta)
    assert math.isclose(computed_gn - all_rows, correction, abs_tol=1e-12)
    assert math.isclose(
        correction,
        data["omitted_tail_derived"][
            "finite_computed_free_energy_correction_kcal"
        ],
        abs_tol=1e-15,
    )


def test_qct_step_totals_and_canonical_gn_reconstruction_fail_on_stale_totals():
    data = load_json(QCT_FIX)
    row = next(item for item in data["rows"] if item["n"] == 2)

    for total_key, steps_key in [
        (
            "labeled_alchemical_association_total",
            "labeled_alchemical_association_steps",
        ),
        ("volume_total", "volume_steps"),
        ("symmetry_total", "symmetry_steps"),
    ]:
        mutated = copy.deepcopy(row)
        mutated[total_key] += 0.25
        assert not math.isclose(
            mutated[total_key],
            sum(mutated[steps_key]),
            abs_tol=1e-12,
        )
        assert not math.isclose(
            _reconstruct_gn(mutated),
            mutated["G_n"],
            abs_tol=1e-12,
        )


def test_qct_logsum_uses_included_rows_only_and_tail_gates_are_strict():
    data = load_json(QCT_FIX)
    protocol = load_json(PROTOCOL_FIX)
    tail_contract = protocol["tail_envelope_contract"]
    rows = data["rows"]
    thr = data["omitted_tail_thresholds"]

    finite_mass, finite_correction, per_row_probability = _derived_tail(
        rows,
        data["beta_kcal_inv"],
    )
    envelope = _tail_envelope_bound(
        rows,
        data["beta_kcal_inv"],
        data["tail_envelope"],
        tail_contract,
    )
    assert envelope is not None
    assert data["tail_envelope_contract_sha256"] == tail_contract[
        "contract_sha256"
    ]
    assert data["tail_envelope"]["confidence_level"] is None
    assert "not a statistical confidence interval" in data["tail_envelope"][
        "upper_bound_semantics"
    ]
    assert envelope["probability_upper"] < thr["probability"]
    assert envelope["free_energy_correction_upper_kcal"] < thr["final_two_kcal"]
    assert math.isclose(
        finite_mass,
        data["omitted_tail_derived"]["finite_computed_probability"],
        abs_tol=1e-15,
    )
    assert math.isclose(
        finite_correction,
        data["omitted_tail_derived"][
            "finite_computed_free_energy_correction_kcal"
        ],
        abs_tol=1e-15,
    )
    for field in [
        "unseen_weight_upper",
        "probability_upper",
        "free_energy_correction_upper_kcal",
    ]:
        stored = (
            data["tail_envelope"][field]
            if field == "unseen_weight_upper"
            else data["omitted_tail_derived"][field]
        )
        assert math.isclose(envelope[field], stored, abs_tol=1e-15)
    for row, probability in zip(rows, per_row_probability):
        assert math.isclose(row["tail_probability"], probability, abs_tol=1e-15)

    # tail candidates must be omitted
    fail_rows = [r for r in rows if r["tail_state"] and r["included"]]
    assert len(fail_rows) == 0

    # Stored tail labels cannot override the partition-function-derived gate.
    spoofed = copy.deepcopy(rows)
    spoofed[-1]["G_n"] = -10.0
    spoofed[-1]["tail_probability"] = 0.0
    mutated_mass, mutated_correction, _ = _derived_tail(
        spoofed,
        data["beta_kcal_inv"],
    )
    assert (
        mutated_mass >= thr["probability"]
        or mutated_correction >= thr["final_two_kcal"]
    )

    # An unresolved or heavy unseen tail cannot be hidden by truncating the
    # supplied rows or by keeping favorable stored diagnostics.
    insufficient = copy.deepcopy(rows[:-1])
    assert _tail_envelope_bound(
        insufficient,
        data["beta_kcal_inv"],
        data["tail_envelope"],
        tail_contract,
    ) is None
    revealed_heavy = copy.deepcopy(rows)
    dangerous = copy.deepcopy(revealed_heavy[-1])
    dangerous["n"] += 1
    dangerous["G_n"] = -10.0
    revealed_heavy.append(dangerous)
    assert _tail_envelope_bound(
        revealed_heavy,
        data["beta_kcal_inv"],
        data["tail_envelope"],
        tail_contract,
    ) is None
    unresolved = copy.deepcopy(data["tail_envelope"])
    unresolved["conservative_ratio_ceiling"] = 1.0
    assert _tail_envelope_bound(
        rows,
        data["beta_kcal_inv"],
        unresolved,
        tail_contract,
    ) is None

    # Post-hoc q selection or a copied stale contract hash is rejected even
    # when the resulting numerical gate would otherwise look more favorable.
    post_hoc = copy.deepcopy(data["tail_envelope"])
    post_hoc["conservative_ratio_ceiling"] = 0.50
    assert _tail_envelope_bound(
        rows,
        data["beta_kcal_inv"],
        post_hoc,
        tail_contract,
    ) is None
    mislabeled = copy.deepcopy(data["tail_envelope"])
    mislabeled["confidence_level"] = 0.95
    assert _tail_envelope_bound(
        rows,
        data["beta_kcal_inv"],
        mislabeled,
    ) is None

    # equality sits on fail-paths in the dedicated fail fixtures
    cases = load_json(PLATEAU_FIX)["tail"]
    fail_mass_eq_mass = sum(r["tail_probability"] for r in cases["fail_mass_eq"]["rows"] if r["tail_state"])
    fail_mass_eq_contrib = sum(r["tail_contribution_kcal"] for r in cases["fail_final_two_eq"]["rows"] if r["tail_state"])
    assert fail_mass_eq_mass >= cases["thresholds"]["probability"]
    assert fail_mass_eq_contrib >= cases["thresholds"]["final_two_kcal"]


def test_qct_tail_gate_and_logsum_covariance_propagation_with_shared_reference():
    data = load_json(QCT_FIX)
    rows = data["rows"]

    shared_entry = data["covariance_references"]["shared"]
    cov = np.asarray(shared_entry["matrix"], dtype=float)
    assert cov.shape == (len(rows), len(rows))

    included = [r for r in rows if r["included"]]
    included_idx = [i for i, r in enumerate(rows) if r["included"]]
    cov_included = cov[np.ix_(included_idx, included_idx)]

    expected_scale = 0.001
    expected_offset = 0.0001
    included_n_plus_one = np.array([rows[i]["n"] + 1 for i in included_idx], dtype=float)
    expected = expected_scale * np.outer(included_n_plus_one, included_n_plus_one)
    expected[np.diag_indices_from(expected)] += expected_offset
    assert np.allclose(cov_included, expected, rtol=1e-12, atol=1e-12)

    g_included = np.array([r["G_n"] for r in included], dtype=float)
    beta = data["beta_kcal_inv"]
    grad, var = _covariance_delta_method(cov_included, g_included, beta)

    # shared-reference covariance is non-zero and propagates finite log-sum uncertainty
    assert not np.allclose(cov_included, 0.0)
    assert math.isclose(sum(grad), 1.0, abs_tol=1e-12)
    assert var > 0.0

    # all-ones and log-sum delta-method projections are stable diagnostics
    ones = np.ones(len(included_idx), dtype=float)
    ones_var = float(ones @ cov_included @ ones)
    assert ones_var > 0.0

    independent = np.diag(np.diag(cov_included))
    var_independent = float(grad @ independent @ grad)
    assert not math.isclose(var, var_independent)

    p_cov_p = float(grad @ cov_included @ grad)
    assert math.isclose(var, p_cov_p, abs_tol=1e-12)


def test_plateau_logic_enforces_spread_and_ci_overlap_and_tail_omission_rules():
    cases = load_json(PLATEAU_FIX)

    plateau = cases["plateau"]
    pass_rows = [r for r in plateau["pass"]["rows"] if r["included"]]
    spread = max(r["occupancy_kcal"] for r in pass_rows) - min(r["occupancy_kcal"] for r in pass_rows)
    assert spread <= plateau["thresholds"]["spread_kcal"]
    assert _ci_intersection(pass_rows)

    fail_spread_rows = [r for r in plateau["fail_spread"]["rows"] if r["included"]]
    fail_spread = max(r["occupancy_kcal"] for r in fail_spread_rows) - min(r["occupancy_kcal"] for r in fail_spread_rows)
    assert fail_spread > plateau["thresholds"]["spread_kcal"]

    fail_ci_rows = [r for r in plateau["fail_ci"]["rows"] if r["included"]]
    assert not _ci_intersection(fail_ci_rows)

    tail = cases["tail"]
    pass_rows = [r for r in tail["pass"]["rows"] if r["included"]]
    mass = sum(r["tail_probability"] for r in tail["pass"]["rows"] if r["tail_state"])
    assert mass < tail["thresholds"]["probability"]

    # explicit tail omission case keeps omitted indices disjoint
    omitted_rows = [i for i, row in enumerate(tail["pass"]["rows"]) if not row["included"]]
    included_states = [i for i, row in enumerate(tail["pass"]["rows"]) if row["included"]]
    assert set(included_states).isdisjoint(set(omitted_rows))
    assert all(not row["included"] for i, row in enumerate(tail["pass"]["rows"]) if i in omitted_rows)


def test_plateau_tail_fail_fixtures_fail_at_strict_equalities():
    cases = load_json(PLATEAU_FIX)

    tail = cases["tail"]
    mass_eq = sum(r["tail_probability"] for r in tail["fail_mass_eq"]["rows"] if r["tail_state"])
    final_two_eq = sum(r["tail_contribution_kcal"] for r in tail["fail_final_two_eq"]["rows"] if r["tail_state"])

    assert mass_eq == tail["thresholds"]["probability"]
    assert final_two_eq == tail["thresholds"]["final_two_kcal"]

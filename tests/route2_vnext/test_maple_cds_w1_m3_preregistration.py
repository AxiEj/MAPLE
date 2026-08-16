from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest

from tools.route2_release import create_maple_cds_w1_m3_preregistration as m3
from tools.route2_release import fit_maple_cds_w1_m3_lad as fitter


def _minimal_preregistration(source_root: Path) -> dict[str, object]:
    payload: dict[str, object] = {key: None for key in m3.PREREGISTRATION_KEYS}
    payload.update(
        {
            "artifact": m3.PREREGISTRATION_ARTIFACT,
            "schema_version": 1,
            "status": "locked-before-first-m3-target-use-or-fit",
            "candidate_profile_id": m3.CANDIDATE_PROFILE_ID,
            "partition": "development-water-only",
            "water_record_count": m3.EXPECTED_WATER_COUNT,
            "source_root": str(source_root),
            "experimental_targets_read_by_m3_preregistration": False,
            "experimental_targets_used_by_subspace_or_folds": False,
            "mnsol_target_table_member_opened_by_m3_preregistration": False,
            "hybrid_prediction_records_read_by_m3_preregistration": False,
            "confirmation_selection_manifest_opened": False,
            "confirmation_records_opened": False,
            "fitting_or_calibration_performed": False,
            "fit_contract": m3.frozen_fit_contract(),
            "development_decision_rule": m3.frozen_development_decision_rule(),
        }
    )
    payload["self_sha256"] = m3.canonical_json_sha256(payload)
    return payload


def _geometry_payload(
    handle: str, atomic_rows: tuple[tuple[int, float, float, float], ...]
) -> bytes:
    lines = [f"{handle} X m062x_mg3s_geom", "0 1"]
    lines.extend(
        f"{atomic_number} {x:.12f} {y:.12f} {z:.12f}"
        for atomic_number, x, y, z in atomic_rows
    )
    return ("\n".join(lines) + "\n").encode()


def test_target_blind_subspace_is_metric_orthogonal_and_unit_invariant():
    generator = np.random.default_rng(20260816)
    matrix = generator.normal(size=(96, 18))
    matrix *= np.linspace(0.2, 3.0, 18)
    stock = np.linspace(-2.0, 4.0, 18)

    result = m3.target_blind_subspace(matrix, stock)
    reduced = np.asarray(result["reduced_design"])
    modes = np.asarray(result["geometry_modes_coefficient_coordinates"])
    norms = np.asarray(result["column_l2_norms"])

    assert reduced.shape == (96, 3)
    assert result["raw_rank"] == 18
    assert result["projected_rank"] == 17
    assert result["reduced_rank"] == 3
    assert modes.T @ np.diag(norms**2) @ modes == pytest.approx(np.eye(2), abs=2.0e-12)
    assert stock @ np.diag(norms**2) @ modes == pytest.approx(np.zeros(2), abs=2.0e-12)
    assert result["positive_unit_reparameterization_prediction_max_abs_error"] < 2.0e-12


def test_target_blind_subspace_rejects_rank_deficiency():
    matrix = np.arange(72.0).reshape(4, 18)
    with pytest.raises(m3.M3PreregistrationError, match="matrix or stock"):
        m3.target_blind_subspace(matrix, np.ones(18))

    matrix = np.eye(18)
    matrix[:, -1] = matrix[:, -2]
    with pytest.raises(m3.M3PreregistrationError, match="rank deficient"):
        m3.target_blind_subspace(matrix, np.ones(18))


def test_geometry_loader_never_opens_target_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = _geometry_payload("a001", ((6, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)))
    second = _geometry_payload("a002", ((8, 0.0, 0.0, 0.0), (1, 0.0, 1.0, 0.0)))
    archive_path = tmp_path / "MNSolDatabase_v2012.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("MNSolDatabase_v2012/MNSol_alldata.txt", b"SECRET TARGET")
        archive.writestr("MNSolDatabase_v2012/all_solutes/a001.xyz", first)
        archive.writestr("MNSolDatabase_v2012/all_solutes/a002.xyz", second)

    original_open = m3.ZipFile.open
    opened: list[str] = []

    def guarded_open(self, name, *args, **kwargs):
        logical = name.filename if isinstance(name, ZipInfo) else str(name)
        opened.append(logical)
        assert not logical.endswith("MNSol_alldata.txt")
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(m3.ZipFile, "open", guarded_open)
    first_sha = hashlib.sha256(first).hexdigest()
    selected = m3.load_selected_geometries_without_table(archive_path, [first_sha])

    assert set(selected) == {first_sha}
    assert selected[first_sha].atomic_numbers == (6, 1)
    assert opened == [
        "MNSolDatabase_v2012/all_solutes/a001.xyz",
        "MNSolDatabase_v2012/all_solutes/a002.xyz",
    ]


def test_heavy_formula_family_is_exact_and_does_not_chain():
    carbon_hydrogen = m3._parse_geometry(
        "c1",
        _geometry_payload("c1", ((6, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0))),
    )
    carbon_more_hydrogen = m3._parse_geometry(
        "c2",
        _geometry_payload(
            "c2",
            (
                (6, 0.0, 0.0, 0.0),
                (1, 1.0, 0.0, 0.0),
                (1, -1.0, 0.0, 0.0),
            ),
        ),
    )
    nitrogen = m3._parse_geometry("n1", _geometry_payload("n1", ((7, 0.0, 0.0, 0.0),)))

    assert m3.heavy_element_formula_family_key(
        carbon_hydrogen
    ) == m3.heavy_element_formula_family_key(carbon_more_hydrogen)
    assert m3.heavy_element_formula_family_key(
        carbon_hydrogen
    ) != m3.heavy_element_formula_family_key(nitrogen)
    assert m3.heavy_element_formula_family_descriptor(carbon_hydrogen) == {
        "contract": "exact-element-count-family-v1",
        "scope": "heavy-atoms",
        "atomic_number_counts": [[6, 1]],
    }


def test_family_fold_assignment_is_deterministic_balanced_and_unsplit():
    stable_ids = [
        hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(31)
    ]
    family_keys: list[str] = []
    for index in range(31):
        family_index = index // 3
        family_keys.append(
            hashlib.sha256(f"family-{family_index}".encode()).hexdigest()
        )

    first = m3.deterministic_family_folds(
        stable_row_ids=stable_ids,
        family_keys=family_keys,
    )
    second = m3.deterministic_family_folds(
        stable_row_ids=stable_ids,
        family_keys=family_keys,
    )

    assert first == second
    assert first["family_count"] == 11
    assert max(first["fold_record_counts"]) - min(first["fold_record_counts"]) <= 3
    row_folds = first["row_fold_assignments"]
    for family in first["families"]:
        assert {row_folds[index] for index in family["member_indices"]} == {
            family["fold"]
        }


def test_family_fold_assignment_fails_when_families_are_too_few():
    stable_ids = [
        hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(10)
    ]
    one_family = ["a" * 64] * 10
    with pytest.raises(m3.M3PreregistrationError, match="fewer"):
        m3.deterministic_family_folds(
            stable_row_ids=stable_ids,
            family_keys=one_family,
        )


def test_unique_lad_recovers_coefficients_and_has_range_certificate():
    generator = np.random.default_rng(1759)
    design = generator.normal(size=(75, 3))
    expected = np.array([0.7, -1.2, 0.35])
    response = design @ expected
    response[::8] += np.linspace(-0.2, 0.2, len(response[::8]))

    result = fitter.fit_numerically_unique_lad(design, response)
    coefficients = np.asarray(result["reduced_coefficients"])

    assert coefficients == pytest.approx(expected, abs=2.0e-12)
    assert result["primary_dual_certificate"]["duality_gap"] < 1.0e-9
    assert all(
        item["range"] <= item["allowed_range"]
        for item in result["coefficient_range_certificates"]
    )


def test_lad_rejects_a_full_rank_but_nonunique_coefficient():
    # gamma_0 is any median in [0, 1], while gamma_1 and gamma_2 are fixed.
    design = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    response = np.array([0.0, 1.0, 2.0, -3.0, -1.0])

    with pytest.raises(fitter.M3FitError, match="numerically non-unique"):
        fitter.fit_numerically_unique_lad(design, response)


def test_family_bootstrap_is_deterministic_and_covers_all_rows():
    errors = np.array([0.2, 0.4, 0.8, 1.6, 0.1])
    families = [[0, 1], [2], [3, 4]]
    first = fitter._cluster_bootstrap_mae(absolute_errors=errors, families=families)
    second = fitter._cluster_bootstrap_mae(absolute_errors=errors, families=families)

    assert first == second
    assert first["replicates"] == m3.BOOTSTRAP_REPLICATES
    with pytest.raises(fitter.M3FitError, match="partition"):
        fitter._cluster_bootstrap_mae(
            absolute_errors=errors, families=[[0, 1], [2], [3]]
        )


@pytest.mark.parametrize("field", ["fit_contract", "development_decision_rule"])
def test_fitter_rejects_a_self_rehashed_preregistration_contract_drift(
    tmp_path: Path, field: str
):
    payload = _minimal_preregistration(m3._SOURCE_ROOT)
    drifted = dict(payload[field])
    drifted["post_hoc_change"] = True
    payload[field] = drifted
    payload["self_sha256"] = m3.canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "self_sha256"}
    )
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    path.chmod(0o444)

    with pytest.raises(fitter.M3FitError, match="contract|decision rule"):
        fitter._read_preregistration(path, m3._SOURCE_ROOT)

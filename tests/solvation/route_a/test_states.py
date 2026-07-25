from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from .conftest import FIXTURE_DIR, load_json

STATE_FIX = FIXTURE_DIR / "state" / "reduced_potential_v1.json"


def _sha64(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{64}", value))


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _row_hash_preimage(table, index):
    return {
        "state_id": table["state_id"],
        "row_index": index,
        "row_label": table["row_labels"][index],
        "lambda_value": table["lambda_values"][index],
        "u_kn_row": table["u_kn"][index],
        "N_k": table["N_k"],
        "frame_ids": table["frame_ids"],
        "beta": table["beta"],
        "units": table["units"],
        "dimensionless": table["dimensionless"],
        "measure_id": table["measure_id"],
        "boundary_conditions": table["boundary_conditions"],
    }


def _table_hash_preimage(table):
    preimage = copy.deepcopy(table)
    preimage.pop("state_hash")
    return preimage


def evaluate_state_contract(table):
    codes = set()

    if table["units"] != "dimensionless":
        codes.add("UNITLESS_MISMATCH")
    if table["dimensionless"] is not True:
        codes.add("UNITLESS_MISMATCH")

    u_kn = table["u_kn"]
    if any(not math.isfinite(v) for row in u_kn for v in row):
        codes.add("REDUCED_POTENTIAL_NONFINITE")

    K = len(table["lambda_values"])
    if len(table["row_labels"]) != K:
        codes.add("STATE_K_MISMATCH")
    if len(table["N_k"]) != K:
        codes.add("STATE_K_MISMATCH")
    if len(table["state_hashes"]) != K:
        codes.add("STATE_HASH_MISMATCH")

    if any(not _sha64(h) for h in table["state_hashes"]) or not _sha64(table["state_hash"]):
        codes.add("STATE_HASH_INVALID")
    elif (
        len(table["state_hashes"]) == K
        and len(table["row_labels"]) == K
        and len(table["u_kn"]) == K
        and len(table["N_k"]) == K
    ):
        expected_rows = [
            _canonical_sha256(_row_hash_preimage(table, index))
            for index in range(K)
        ]
        if table["state_hashes"] != expected_rows:
            codes.add("STATE_HASH_MISMATCH")
        if table["state_hash"] != _canonical_sha256(_table_hash_preimage(table)):
            codes.add("STATE_HASH_MISMATCH")

    if len(u_kn) != K or any(len(row) != len(table["frame_ids"]) for row in u_kn):
        codes.add("STATE_MATRIX_SHAPE_MISMATCH")

    if len(table["frame_ids"]) != sum(table["N_k"]):
        codes.add("STATE_MATRIX_SHAPE_MISMATCH")

    if not table["measure_id"].strip():
        codes.add("STATE_MEASURE_MISSING")

    bc = table["boundary_conditions"]
    if bc.get("pbc", True):
        codes.add("STATE_BOUNDARY_MISMATCH")
    if "box_A" in bc and bc["box_A"] is not None:
        codes.add("STATE_BOUNDARY_MISMATCH")

    return sorted(codes)


def test_state_contract_shape_units_and_nonperiodic_boundary_fixture_is_strict():
    table = load_json(STATE_FIX)

    codes = evaluate_state_contract(table)
    assert codes == []

    bc = table["boundary_conditions"]
    assert bc["pbc"] is False
    assert "box" not in bc
    assert "box_A" not in bc

    K = len(table["lambda_values"])
    N = len(table["frame_ids"])
    assert K == len(table["row_labels"]) == len(table["N_k"]) == len(table["state_hashes"]) == len(table["u_kn"])
    assert N == len(table["frame_ids"]) == sum(table["N_k"])
    assert all(len(row) == N for row in table["u_kn"])


def test_state_contract_mutation_codes_are_stable_for_key_mismatches():
    table = load_json(STATE_FIX)

    # units / dimensionless mismatch
    unit_mut = copy.deepcopy(table)
    unit_mut["units"] = "kcal/mol"
    unit_mut["dimensionless"] = False
    assert evaluate_state_contract(unit_mut) == [
        "STATE_HASH_MISMATCH",
        "UNITLESS_MISMATCH",
    ]

    # non-finite reduced potential
    nonfinite = copy.deepcopy(table)
    nonfinite["u_kn"][0][0] = float("nan")
    assert evaluate_state_contract(nonfinite) == [
        "REDUCED_POTENTIAL_NONFINITE",
        "STATE_HASH_MISMATCH",
    ]

    # K mismatch
    k_mut = copy.deepcopy(table)
    k_mut["lambda_values"] = k_mut["lambda_values"][:-1]
    assert "STATE_K_MISMATCH" in evaluate_state_contract(k_mut)
    assert "STATE_MATRIX_SHAPE_MISMATCH" in evaluate_state_contract(k_mut)

    # N mismatch
    n_mut = copy.deepcopy(table)
    n_mut["frame_ids"] = n_mut["frame_ids"][:-1]
    assert evaluate_state_contract(n_mut) == [
        "STATE_HASH_MISMATCH",
        "STATE_MATRIX_SHAPE_MISMATCH",
    ]

    # Invalid syntax remains distinct from a well-formed but stale hash.
    hash_mut = copy.deepcopy(table)
    hash_mut["state_hashes"][0] = "zz"
    codes = evaluate_state_contract(hash_mut)
    assert "STATE_HASH_INVALID" in codes

    for field, mutate in [
        ("lambda", lambda item: item["lambda_values"].__setitem__(0, 0.05)),
        ("u_kn", lambda item: item["u_kn"][0].__setitem__(0, 0.25)),
        ("measure", lambda item: item.__setitem__("measure_id", "other-measure")),
        (
            "boundary",
            lambda item: item.__setitem__(
                "boundary_conditions",
                {"pbc": False, "reference": "mutated"},
            ),
        ),
    ]:
        stale_hash = copy.deepcopy(table)
        mutate(stale_hash)
        assert "STATE_HASH_MISMATCH" in evaluate_state_contract(stale_hash), field

    # measure mismatch
    measure_mut = copy.deepcopy(table)
    measure_mut["measure_id"] = ""
    assert evaluate_state_contract(measure_mut) == [
        "STATE_HASH_MISMATCH",
        "STATE_MEASURE_MISSING",
    ]

    # boundary mismatch
    boundary_mut = copy.deepcopy(table)
    boundary_mut["boundary_conditions"] = {"pbc": True, "box_A": [[10,0,0],[0,10,0],[0,0,10]]}
    assert evaluate_state_contract(boundary_mut) == [
        "STATE_BOUNDARY_MISMATCH",
        "STATE_HASH_MISMATCH",
    ]

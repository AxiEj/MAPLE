from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import pytest

from maple.function.route2_solvents import (
    ROUTE2_SMD_SOLVENTS,
    SMD_SOLVENTS,
    SUPPORTED_ROUTE2_SMD_SOLVENTS,
    SUPPORTED_SMD_SOLVENTS,
    normalize_route2_solvent_name,
    route2_solvent_spec,
)
from maple.function.smd_solvent_data import (
    PYSCF_SMD_SOURCE_MODULE_SHA256,
    PYSCF_SMD_SOURCE_VERSION,
    PYSCF_SMD_TABLE_SHA256,
    SMD_SOLVENT_DESCRIPTOR_SNAPSHOT,
)

_HISTORICAL_SPECS = {
    "water": ("water", "water", (1.3328, 1.3323, 0.82, 0.35, -1.0, 78.355, -1.0, -1.0)),
    "methanol": (
        "methanol",
        "methanol",
        (1.3288, 1.3265, 0.43, 0.47, 31.77, 32.613, 0.0, 0.0),
    ),
    "ethanol": (
        "ethanol",
        "ethanol",
        (1.3611, 1.3593, 0.37, 0.48, 31.62, 24.852, 0.0, 0.0),
    ),
    "acetonitrile": (
        "acetonitrile",
        "acetonitrile",
        (1.3442, 1.3416, 0.07, 0.32, 41.25, 35.688, 0.0, 0.0),
    ),
    "dimethylsulfoxide": (
        "dimethylsulfoxide",
        "dimethylsulfoxide",
        (1.4783, 1.4783, 0.0, 0.88, 61.78, 46.826, 0.0, 0.0),
    ),
    "dimethylformamide": (
        "N,N-dimethylformamide",
        "dimethylformamide",
        (1.4305, 1.4280, 0.0, 0.74, 49.56, 37.219, 0.0, 0.0),
    ),
    "tetrahydrofuran": (
        "tetrahydrofuran",
        "tetrahydrofuran",
        (1.4050, 1.4044, 0.0, 0.48, 39.44, 7.4257, 0.0, 0.0),
    ),
    "chloroform": (
        "chloroform",
        "chloroform",
        (1.4459, 1.4431, 0.15, 0.02, 38.39, 4.7113, 0.0, 0.75),
    ),
    "dichloromethane": (
        "dichloromethane",
        "methylenechloride",
        (1.4242, 1.4212, 0.10, 0.05, 39.15, 8.93, 0.0, 0.667),
    ),
    "toluene": (
        "toluene",
        "toluene",
        (1.4961, 1.4936, 0.0, 0.14, 40.2, 2.3741, 0.857, 0.0),
    ),
    "hexane": (
        "n-hexane",
        "hexane",
        (1.3749, 1.3722, 0.0, 0.0, 25.75, 1.8819, 0.0, 0.0),
    ),
}


def test_full_catalog_and_historical_profile_scope_are_separate_and_immutable():
    assert len(SMD_SOLVENTS) == len(SUPPORTED_SMD_SOLVENTS) == 179
    assert len(ROUTE2_SMD_SOLVENTS) == len(SUPPORTED_ROUTE2_SMD_SOLVENTS) == 11
    assert SUPPORTED_ROUTE2_SMD_SOLVENTS == frozenset(_HISTORICAL_SPECS)
    assert SUPPORTED_ROUTE2_SMD_SOLVENTS < SUPPORTED_SMD_SOLVENTS
    with pytest.raises(TypeError):
        SMD_SOLVENTS["water"] = route2_solvent_spec("water")


@pytest.mark.parametrize(("canonical", "expected"), sorted(_HISTORICAL_SPECS.items()))
def test_historical_specs_remain_byte_for_byte_equivalent(canonical, expected):
    pyscf_name, mnsol_name, descriptors = expected
    spec = route2_solvent_spec(canonical)
    assert ROUTE2_SMD_SOLVENTS[canonical] is spec
    assert spec.name == canonical
    assert spec.pyscf_smd_name == pyscf_name
    assert spec.mnsol_name == mnsol_name
    assert spec.descriptors.as_pyscf_tuple() == descriptors


def test_new_catalog_entries_do_not_claim_unverified_mnsol_join_keys():
    unmapped = SUPPORTED_SMD_SOLVENTS - SUPPORTED_ROUTE2_SMD_SOLVENTS
    assert len(unmapped) == 168
    assert all(SMD_SOLVENTS[name].mnsol_name is None for name in unmapped)
    assert route2_solvent_spec("2-propanol").mnsol_name is None
    assert route2_solvent_spec("ethyl acetate").mnsol_name is None


def test_every_catalog_record_has_valid_identity_and_published_descriptors():
    parser_safe = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
    assert len(SMD_SOLVENT_DESCRIPTOR_SNAPSHOT) == 179
    assert {row[0] for row in SMD_SOLVENT_DESCRIPTOR_SNAPSHOT} == set(SMD_SOLVENTS)
    assert len({row[1] for row in SMD_SOLVENT_DESCRIPTOR_SNAPSHOT}) == 179

    for canonical, pyscf_name, values in SMD_SOLVENT_DESCRIPTOR_SNAPSHOT:
        spec = SMD_SOLVENTS[canonical]
        assert parser_safe.fullmatch(canonical)
        assert pyscf_name
        assert len(values) == 8
        assert all(math.isfinite(value) for value in values)
        assert any(value != 0 for value in values)
        assert values[0] > 0
        assert values[5] > 0
        assert spec.pyscf_smd_name == pyscf_name
        assert spec.descriptors.as_pyscf_tuple() == values
        assert normalize_route2_solvent_name(canonical) == canonical
        assert normalize_route2_solvent_name(pyscf_name) == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("H2O", "water"),
        ("DMF", "dimethylformamide"),
        ("n-hexane", "hexane"),
        ("ethyl acetate", "ethylethanoate"),
        ("methyl acetate", "methylethanoate"),
        ("ethylene glycol", "1-2-ethanediol"),
        ("isopropanol", "2-propanol"),
        ("isopropyl alcohol", "2-propanol"),
        ("n-octanol", "1-octanol"),
        ("decalin (cis/trans mixture)", "decalin-cis-trans-mixture"),
        ("cis-decalin", "cis-decalin"),
        ("trans-decalin", "trans-decalin"),
        ("xylene (mixture)", "xylene-mixture"),
        ("m-xylene", "m-xylene"),
        ("o-xylene", "o-xylene"),
        ("p-xylene", "p-xylene"),
    ],
)
def test_unambiguous_aliases_preserve_isomers_and_mixtures(alias, canonical):
    assert normalize_route2_solvent_name(alias) == canonical
    assert route2_solvent_spec(alias).name == canonical


@pytest.mark.parametrize("ambiguous", ["decalin", "xylene"])
def test_ambiguous_family_names_are_not_guessed(ambiguous):
    with pytest.raises(ValueError, match="Unsupported Route 2 SMD solvent"):
        route2_solvent_spec(ambiguous)


@pytest.mark.parametrize("invalid", ["", "   ", None, "made-up-solvent"])
def test_empty_and_unknown_solvent_names_fail_closed(invalid):
    with pytest.raises(ValueError, match="Unsupported Route 2 SMD solvent"):
        route2_solvent_spec(invalid)


def test_snapshot_matches_exact_pinned_pyscf_table_and_source_hash():
    pyscf = pytest.importorskip("pyscf")
    assert str(pyscf.__version__) == PYSCF_SMD_SOURCE_VERSION
    from pyscf.solvent import smd

    upstream = {name: tuple(values) for name, values in smd.solvent_db.items() if name}
    snapshot = {
        pyscf_name: values
        for _canonical, pyscf_name, values in SMD_SOLVENT_DESCRIPTOR_SNAPSHOT
    }
    assert snapshot == upstream
    assert "" in smd.solvent_db
    assert tuple(smd.solvent_db[""]) == (0, 0, 0, 0, 0, 0, 0, 0)

    source_path = Path(inspect.getfile(smd))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        PYSCF_SMD_SOURCE_MODULE_SHA256
    )
    normalized = json.dumps(
        upstream,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    assert hashlib.sha256(normalized).hexdigest() == PYSCF_SMD_TABLE_SHA256


def test_runtime_catalog_import_does_not_import_pyscf():
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from maple.function.route2_solvents import SMD_SOLVENTS; "
                "assert len(SMD_SOLVENTS) == 179; "
                "assert 'pyscf' not in sys.modules"
            ),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

from __future__ import annotations

import math

from .conftest import FIXTURE_DIR, load_json


SYM_FIX = FIXTURE_DIR / "symmetry" / "symmetry.json"


def _rtln(n: int, t: float, r: float) -> float:
    return r * t * math.log(float(n))


def test_per_edge_symmetry_matches_fixture():
    spec = load_json(SYM_FIX)
    t = spec["temperature_k"]
    r = spec["gas_constant_kcal_per_mol_k"]

    for n_str, expected in spec["expected_rtln"].items():
        n = int(n_str)
        observed = _rtln(n, t, r)
        assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-10)


def test_cumulative_factorial_symmetry():
    spec = load_json(SYM_FIX)
    t = spec["temperature_k"]
    r = spec["gas_constant_kcal_per_mol_k"]

    n = spec["cumulative_factorial"]["n"]
    observed = _rtln(math.factorial(n), t, r)
    assert math.isclose(observed, spec["cumulative_factorial"]["rtln"], rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(1.0 / math.factorial(n), math.exp(-observed / (r * t)), rel_tol=0.0, abs_tol=1e-12)

#!/usr/bin/env python3
"""Subsample-bias audit of the frozen legacy exact-GTO confirmation-123 run.

The frozen artifact
``route2-legacy-exact-gto-confirmation123-diagnostic-v1.json`` publishes an
aggregate ``mae_kcal_mol`` over the 77 records that produced an energy, and
separately records that 46 of 123 records were fail-closed because PCMSolver
warned for every cavity attempt under policy
``intrinsic-smd-probe0-noaddsph-v1``.

This script tests whether those 46 rejections are independent of solute class.
They are not.  The fail-closed rate rises monotonically with size bin and is
strongly element-class dependent, while the surviving MAE also rises with size
bin.  The published aggregate is therefore computed on a subsample enriched in
the easy records, and understates the panel error.

The script reads only the frozen artifact, uses only the Python standard
library, and writes nothing.  It selects no record, changes no threshold, and
produces no accuracy claim: it is an audit of an existing published aggregate.

Usage:
    python3 analyze_confirmation123_subsample_bias.py [artifact.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ARTIFACT = "route2-legacy-exact-gto-confirmation123-diagnostic-v1.json"
DIAGNOSTIC_KEY = "legacy_exact_gto_diagnostic"
ERROR_KEY = "absolute_error_kcal_mol"

# Bins are read verbatim from the frozen records; none is defined here.
BIN_KEYS = ("size_bin", "flexibility_bin", "element_class", "heteroatom_bin")

# Upper 5% points of the chi-square distribution, for reference only.
CHI2_CRITICAL_5PCT = {1: 3.841, 2: 5.991, 3: 7.815}


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    records = artifact["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} contains no record list.")
    return records


def split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition into published-energy records and fail-closed records."""
    published, fail_closed = [], []
    for record in records:
        diagnostic = record.get(DIAGNOSTIC_KEY, {})
        (published if ERROR_KEY in diagnostic else fail_closed).append(record)
    return published, fail_closed


def chi_square_independence(
    records: list[dict],
    published: list[dict],
    fail_closed: list[dict],
    key: str,
) -> tuple[float, int]:
    """Pearson chi-square for independence of fail-closed status and ``key``."""
    bins = sorted({record[key] for record in records})
    observed = {value: [0, 0] for value in bins}
    for record in published:
        observed[record[key]][0] += 1
    for record in fail_closed:
        observed[record[key]][1] += 1

    total = len(records)
    column_totals = (len(published), len(fail_closed))
    statistic = 0.0
    for value in bins:
        row_total = sum(observed[value])
        for column, column_total in enumerate(column_totals):
            expected = row_total * column_total / total
            if expected > 0.0:
                statistic += (observed[value][column] - expected) ** 2 / expected
    return statistic, len(bins) - 1


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stratified_estimate(
    records: list[dict],
    published: list[dict],
    key: str,
) -> tuple[float, dict[str, tuple[int, int, float]]]:
    """Re-weight the surviving MAE by the *full* panel's bin populations.

    Each fail-closed record is imputed the mean absolute error of the surviving
    records in its own bin.  This is deliberately conservative: a record that
    defeats the cavity builder is unlikely to be an average member of its bin.
    """
    errors = defaultdict(list)
    for record in published:
        errors[record[key]].append(record[DIAGNOSTIC_KEY][ERROR_KEY])

    populations = defaultdict(int)
    for record in records:
        populations[record[key]] += 1

    detail: dict[str, tuple[int, int, float]] = {}
    weighted, counted = 0.0, 0
    for value in sorted(populations):
        if value not in errors:
            continue
        bin_mae = mean(errors[value])
        population = populations[value]
        detail[value] = (len(errors[value]), population, bin_mae)
        weighted += population * bin_mae
        counted += population
    return weighted / counted, detail


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / ARTIFACT
    records = load_records(path)
    published, fail_closed = split(records)

    print(f"artifact          : {path.name}")
    print(f"records           : {len(records)}")
    print(f"published energy  : {len(published)}")
    print(f"fail-closed       : {len(fail_closed)}"
          f"  ({len(fail_closed) / len(records):.1%})")
    print(f"published MAE     : {mean([r[DIAGNOSTIC_KEY][ERROR_KEY] for r in published]):.4f}"
          " kcal/mol  (frozen artifact aggregate)")

    print("\n== Is fail-closed status independent of solute class? ==")
    for key in BIN_KEYS:
        statistic, dof = chi_square_independence(records, published, fail_closed, key)
        critical = CHI2_CRITICAL_5PCT.get(dof)
        verdict = "dependent" if critical and statistic > critical else "not resolved"
        print(f"  {key:16s} chi2={statistic:6.2f}  dof={dof}"
              f"  crit(5%)={critical}  -> {verdict}")

    print("\n== Fail-closed rate and surviving MAE, by bin ==")
    for key in BIN_KEYS:
        counts_ok: dict[str, int] = defaultdict(int)
        counts_bad: dict[str, int] = defaultdict(int)
        errors: dict[str, list[float]] = defaultdict(list)
        for record in published:
            counts_ok[record[key]] += 1
            errors[record[key]].append(record[DIAGNOSTIC_KEY][ERROR_KEY])
        for record in fail_closed:
            counts_bad[record[key]] += 1
        print(f"  --- {key} ---")
        for value in sorted(set(counts_ok) | set(counts_bad)):
            good, bad = counts_ok[value], counts_bad[value]
            rate = bad / (good + bad)
            mae = mean(errors[value]) if errors[value] else float("nan")
            print(f"    {value:20s} ok={good:3d} fail={bad:3d}"
                  f"  fail_rate={rate:.2f}  surviving_MAE={mae:.3f}")

    print("\n== Conservative stratified re-estimate of the panel MAE ==")
    for key in ("size_bin", "element_class"):
        estimate, detail = stratified_estimate(records, published, key)
        print(f"  stratified by {key}: MAE ~ {estimate:.4f} kcal/mol")
        for value, (n_ok, population, bin_mae) in detail.items():
            print(f"      {value:20s} survivors={n_ok:3d} panel={population:3d}"
                  f"  bin_MAE={bin_mae:.3f}")

    print("\nNote: the protocol's confirmation gate additionally requires"
          " failure_rate == 0.\n      A 37.4% fail-closed rate fails that gate"
          " independently of any accuracy value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

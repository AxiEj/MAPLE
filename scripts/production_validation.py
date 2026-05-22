#!/usr/bin/env python
"""Run the PBC-MD release acceptance matrix and emit a dated report (WS3).

This is the non-skippable production-validation ship gate. By default it runs
backend-free against a Lennard-Jones reference calculator (a real conservative
potential with stress); pass --model to validate a real MAPLE ML backend.

Examples
--------
    # Backend-free harness self-validation:
    python scripts/production_validation.py

    # Validate a real periodic backend:
    python scripts/production_validation.py --model mace-mp-pbc-small --device cuda

The report (markdown + JSON) and per-run provenance manifests are written under
validation/reports/ (override with --outdir). This is the single release gate: the
exit code is 0 only if every acceptance class passes AND none is skipped
(inconclusive). A fired barostat clamp or an unknown neighbor cutoff already fails the
matrix (via the barostat_clamp_free class and the MD cutoff-admission gate), and the
per-run provenance manifests record the unit contract and cutoff policy.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from maple.function.dispatcher.md.validation import (  # noqa: E402
    lj_reference_factory,
    load_thresholds,
    run_acceptance_matrix,
    write_report,
)


def _real_model_factory(model: str, device: str | None, output: str):
    """Build a real MAPLE calculator once and reuse it across acceptance classes."""
    import torch

    from maple.function.calculator import SetCalculator
    from maple.function.dispatcher.md.provenance import collect_calculator_provenance

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    calc = SetCalculator(resolved_device, model, output, model_options={}).set_calculator()
    try:
        calc.maple_provenance = collect_calculator_provenance(
            calc, model=model, device=resolved_device
        )
    except Exception:
        pass
    return lambda: calc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="MAPLE model name (default: built-in LJ reference calculator)")
    parser.add_argument("--device", default=None, help="torch device (cpu/cuda)")
    parser.add_argument("--thresholds", default=None, help="path to thresholds.toml")
    parser.add_argument("--outdir", default=str(_REPO_ROOT / "validation" / "reports"),
                        help="report output directory")
    parser.add_argument("--quick", action="store_true",
                        help="shorter runs (smoke; not for a real ship gate)")
    args = parser.parse_args(argv)

    thresholds = load_thresholds(args.thresholds)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.model:
        factory = _real_model_factory(args.model, args.device, str(outdir / "calc.out"))
        label = args.model
    else:
        factory = lj_reference_factory()
        label = "lj-reference"

    results = run_acceptance_matrix(factory, thresholds, quick=args.quick)

    sample_calc = factory()
    calc_contract = {
        "energy_unit": getattr(sample_calc, "maple_energy_unit", None),
        "force_unit": getattr(sample_calc, "maple_force_unit", None),
        "stress_unit": getattr(sample_calc, "maple_stress_unit", None),
        "neighbor_cutoff_A": (
            getattr(sample_calc, "neighbor_cutoff_A", None)
            or getattr(sample_calc, "maple_neighbor_cutoff", None)
        ),
    }
    extra = {
        "calculator_contract": calc_contract,
        # The release gate runs strict: an unknown cutoff is rejected at MD admission.
        "cutoff_policy": {"allow_unknown_cutoff": False},
    }
    report = write_report(results, thresholds, outdir, calculator_label=label, extra=extra)

    n_pass = sum(1 for r in results if r.passed)
    n_skip = sum(1 for r in results if r.status == "skip")
    n_fail = sum(1 for r in results if not r.passed and r.status != "skip")
    clamp_count = next(
        (r.metrics.get("clamp_count") for r in results if r.name == "barostat_clamp_free"),
        None,
    )
    # Ship gate: every class passes AND none is inconclusive (a skip is not a pass).
    ok = n_fail == 0 and n_skip == 0 and all(r.passed for r in results)

    print(f"\nMD acceptance matrix ({label}): {'PASS' if ok else 'FAIL'}")
    for r in results:
        print(f"  [{r.status.upper():4}] {r.name}: {r.detail}")
    print(
        f"\nsummary: {n_pass} pass, {n_fail} fail, {n_skip} skip; "
        f"barostat clamps={clamp_count}; "
        f"units={calc_contract['energy_unit']}/{calc_contract['force_unit']}/"
        f"{calc_contract['stress_unit']}"
    )
    print(f"Report: {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

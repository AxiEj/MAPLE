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
validation/reports/ (override with --outdir). Exit code is 0 only if every
acceptance class passes.
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
    report = write_report(results, thresholds, outdir, calculator_label=label)

    all_passed = all(r.passed for r in results)
    print(f"\nMD acceptance matrix ({label}): {'PASS' if all_passed else 'FAIL'}")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    print(f"\nReport: {report}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

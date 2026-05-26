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
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from maple.function.dispatcher.md.validation import (  # noqa: E402
    lj_reference_factory,
    load_smoke_thresholds,
    load_thresholds,
    run_acceptance_matrix,
    validation_system_summary,
    write_report,
)


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?$"
)


def _coerce_model_option_value(value: str):
    text = value.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text) and any(char in text for char in ".eE"):
        return float(text)
    return value


def _parse_model_options(items: list[str] | None) -> dict:
    parsed = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--model-option must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--model-option key must be non-empty, got {item!r}")
        if key in parsed:
            raise ValueError(f"duplicate --model-option key: {key}")
        parsed[key] = _coerce_model_option_value(value)
    return parsed


def _apply_validation_model_defaults(model: str | None, model_options: dict) -> dict:
    """Return model options appropriate for production validation.

    The MAPLE runtime default for MACE/MACE-Polar PBC remains float32 for speed.
    MACE-Polar's finite-difference stress gate is precision-sensitive enough
    that float32 can quantize small strain-energy differences to zero, so
    validate PolarMACE in float64 unless the caller explicitly chooses a dtype.
    """
    options = dict(model_options)
    if model in {"macepol-pbc-small", "macepol-pbc-medium", "macepol-pbc-large"}:
        options.setdefault("default_dtype", "float64")
    return options


def _real_model_factory(model: str, device: str | None, output: str, model_options: dict):
    """Build a real MAPLE calculator once and reuse it across acceptance classes."""
    import torch

    from maple.function.calculator import SetCalculator
    from maple.function.dispatcher.md.provenance import collect_calculator_provenance

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    calc = SetCalculator(
        resolved_device, model, output, model_options=model_options
    ).set_calculator()
    try:
        calc.maple_provenance = collect_calculator_provenance(
            calc, model=model, device=resolved_device, model_options=model_options
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
    parser.add_argument(
        "--model-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "repeatable calculator option; true/false -> bool, integer -> int, "
            "float/scientific -> float, otherwise string"
        ),
    )
    args = parser.parse_args(argv)

    try:
        model_options = _parse_model_options(args.model_option)
    except ValueError as exc:
        parser.error(str(exc))
    model_options = _apply_validation_model_defaults(args.model, model_options)

    if args.thresholds:
        thresholds = load_thresholds(args.thresholds)
    else:
        thresholds = load_smoke_thresholds() if args.quick else load_thresholds()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.model:
        factory = _real_model_factory(
            args.model, args.device, str(outdir / "calc.out"), model_options
        )
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
        "local_descriptor_cutoff_A": getattr(sample_calc, "local_descriptor_cutoff_A", None),
        "short_range_realspace_cutoff_A": getattr(
            sample_calc, "short_range_realspace_cutoff_A", None
        ),
        "long_range_coulomb_cutoff_A": (
            getattr(sample_calc, "long_range_coulomb_cutoff_A", None)
            or getattr(sample_calc, "lrcoulomb_cutoff_A", None)
        ),
        "long_range_method": (
            getattr(sample_calc, "lrcoulomb_method", None)
            or (getattr(sample_calc, "maple_model_options", {}) or {}).get("coulomb")
            or "none"
        ),
    }
    extra = {
        "calculator_contract": calc_contract,
        # The release gate runs strict: an unknown cutoff is rejected at MD admission.
        "cutoff_policy": {"allow_unknown_cutoff": False},
        "validation_mode": "quick-smoke" if args.quick else "production-validation",
        "production_validated": not args.quick,
        "model_options": model_options,
        "validation_system": validation_system_summary(factory),
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

    mode = "quick smoke" if args.quick else "production validation"
    print(f"\nMD acceptance matrix ({label}, {mode}): {'PASS' if ok else 'FAIL'}")
    if args.quick:
        print("  NOTE: --quick is compatibility smoke only; it is not production-validated.")
    for r in results:
        print(f"  [{r.status.upper():4}] {r.name}: {r.detail}")
    print(
        f"\nsummary: {n_pass} pass, {n_fail} fail, {n_skip} skip; "
        f"barostat clamps={clamp_count}; "
        f"units={calc_contract['energy_unit']}/{calc_contract['force_unit']}/"
        f"{calc_contract['stress_unit']}"
    )
    import json

    report_payload = json.loads(report.with_suffix(".json").read_text())
    print(f"Report: {report}")
    print(f"Report SHA256: {report_payload.get('markdown_report_sha256')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

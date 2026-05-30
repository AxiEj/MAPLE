#!/usr/bin/env python
"""Run the PBC-MD release acceptance matrix and emit a dated report (WS3).

This is the non-skippable, single-target production-validation gate. By default
it runs backend-free against a Lennard-Jones reference calculator (a real
conservative potential with stress); pass --model to validate one real MAPLE ML
backend target. A production backend claim requires one passing report for every
target in validation/required_pbc_backends.toml, followed by the aggregate
checker.

Examples
--------
    # Backend-free harness self-validation:
    python scripts/production_validation.py

    # Validate one real periodic backend target:
    python scripts/production_validation.py --model mace-mp-pbc-small --device cuda

    # Validate one AIMNet2 long-range mode target:
    python scripts/production_validation.py --model aimnet2-pbc --device cuda --model-option coulomb=ewald

    # After all real-backend targets have reports, verify the aggregate claim:
    python scripts/check_production_backend_matrix.py

The report (markdown + JSON) and per-run provenance manifests are written under
validation/reports/<artifact_id>/ (override with --outdir/--workdir):
report.md, report.json, and runs/*_md_{manifest.json,summary.txt,thermo.dat}. This
single-target gate exits 0 only if every acceptance class passes, none is skipped
(inconclusive), the report is from the current clean git commit, and no barostat
clamp fired. An unknown neighbor cutoff already fails the matrix (via the MD
cutoff-admission gate), and the per-run provenance manifests record the unit
contract and cutoff policy.
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
    make_report_context,
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
    """Build a fresh real MAPLE calculator for each independent validation probe."""
    import torch

    from maple.function.calculator import SetCalculator
    from maple.function.dispatcher.md.provenance import collect_calculator_provenance

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    def factory():
        # Official periodic backends may cache cell/neighbor-list/long-range
        # state below ASE's reset boundary.  Production validation is an
        # evidence artifact, so isolate every independent acceptance class and
        # finite-difference probe with a fresh calculator instead of letting a
        # long NPT trajectory contaminate a later stress or H-cons check.
        calc = SetCalculator(
            resolved_device, model, output, model_options=model_options
        ).set_calculator()
        try:
            calc.maple_provenance = collect_calculator_provenance(
                calc, model=model, device=resolved_device, model_options=model_options
            )
        except Exception:
            pass
        return calc

    factory.maple_model_name = model
    factory.maple_model_options = dict(model_options)
    return factory


def _validation_target(label: str, model_options: dict, calc_contract: dict) -> dict:
    """Describe the exact model+options scope covered by this report."""
    options = dict(model_options)
    method = calc_contract.get("long_range_method")
    if label.startswith("aimnet2") and method:
        options.setdefault("coulomb", method)
        if method == "dsf" and calc_contract.get("long_range_coulomb_cutoff_A") is not None:
            options.setdefault("cutoff", calc_contract["long_range_coulomb_cutoff_A"])
    if options:
        opts = ", ".join(f"{key}={options[key]!r}" for key in sorted(options))
        target = f"{label} ({opts})"
    else:
        target = label
    return {
        "model": label,
        "model_options": options,
        "target": target,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=None,
                        help="MAPLE model name (default: built-in LJ reference calculator)")
    parser.add_argument("--device", default=None, help="torch device (cpu/cuda)")
    parser.add_argument("--thresholds", default=None, help="path to thresholds.toml")
    parser.add_argument("--outdir", default=str(_REPO_ROOT / "validation" / "reports"),
                        help="report output directory")
    parser.add_argument(
        "--workdir",
        default=None,
        help=(
            "persistent directory for per-run MD artifacts; default is "
            "<outdir>/<artifact_id>/runs"
        ),
    )
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
    report_context = make_report_context(thresholds)
    artifact_id = report_context["artifact_id"]
    artifact_dir = outdir / artifact_id
    run_workdir = Path(args.workdir) if args.workdir else artifact_dir / "runs"
    run_workdir.mkdir(parents=True, exist_ok=True)

    if args.model:
        factory = _real_model_factory(
            args.model, args.device, str(run_workdir / "calc.out"), model_options
        )
        label = args.model
    else:
        factory = lj_reference_factory()
        label = "lj-reference"

    results = run_acceptance_matrix(
        factory,
        thresholds,
        run_workdir,
        quick=args.quick,
        validation_artifact_id=artifact_id,
    )

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
        "production_validation_candidate": not args.quick,
        "model_options": model_options,
        "validation_target": _validation_target(label, model_options, calc_contract),
        "validation_system": validation_system_summary(factory),
    }
    report = write_report(
        results,
        thresholds,
        outdir,
        calculator_label=label,
        extra=extra,
        artifact_id=artifact_id,
        generated_utc=report_context["generated_utc"],
        environment=report_context["environment"],
        run_workdir=run_workdir,
        artifact_dir=artifact_dir,
    )

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
    if not args.quick and not report_payload.get("production_validated", False):
        failed = report_payload.get("release_gate", {}).get("failed_criteria", [])
        if failed:
            print("Release gate failed criteria:")
            for criterion in failed:
                print(f"  - {criterion}")
    return 0 if (ok if args.quick else report_payload.get("production_validated", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

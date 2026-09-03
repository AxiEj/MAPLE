#!/usr/bin/env python3
"""Audit the supplied MACE-POLAR-EF-v2 checkpoint on a water field canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)

POSITIONS_ANGSTROM = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]],
    dtype=float,
)
FIELD_STEPS_EV_PER_E_ANGSTROM = (5.0e-4, 1.0e-3, 2.0e-3)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = MACEPolarEFConfig(
        checkpoint_path=str(arguments.checkpoint),
        atomic_numbers=(8, 1, 1),
        total_charge=0,
        spin_multiplicity=1,
    )
    model = MACEPolarEFEnergyModel(config)
    audits = [
        model.audit_uniform_field_passivity(
            POSITIONS_ANGSTROM,
            field_step_ev_per_e_angstrom=step,
        )
        for step in FIELD_STEPS_EV_PER_E_ANGSTROM
    ]
    audit = audits[1]
    passivity_passed = all(item.passivity_passed for item in audits)
    resolution_scan = [
        {
            "field_step_ev_per_e_angstrom": item.field_step_ev_per_e_angstrom,
            "energy_hessian_eigenvalues": (item.energy_hessian_eigenvalues.tolist()),
            "maximum_positive_curvature": item.maximum_positive_curvature,
            "passivity_passed": item.passivity_passed,
        }
        for item in audits
    ]
    payload = {
        "artifact": "mace-polar-ef-v2-water-passivity-audit-v1",
        "model": config.as_provenance(),
        "runtime": dict(model.runtime_provenance),
        "positions_angstrom": POSITIONS_ANGSTROM.tolist(),
        "field_step_ev_per_e_angstrom": (audit.field_step_ev_per_e_angstrom),
        "energy_hessian": audit.energy_hessian.tolist(),
        "energy_hessian_eigenvalues": (audit.energy_hessian_eigenvalues.tolist()),
        "maximum_positive_curvature": audit.maximum_positive_curvature,
        "tolerance": audit.tolerance,
        "passivity_passed": passivity_passed,
        "resolution_scan": resolution_scan,
        "ddpcm_coupling_admitted": passivity_passed,
        "provenance": dict(audit.provenance),
    }
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_RECORD_COUNT = 505
PREREG_PATH = Path('/tmp/maple-route2-hybrid-smd-development-prereg-v2.json')
RUNNER_PATH = Path('/tmp/run_hybrid_smd_development_v2.py')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = np.asarray([row['predicted_delta_g_kcal_mol'] for row in records])
    experimental = np.asarray([row['experimental_delta_g_kcal_mol'] for row in records])
    signed = predicted - experimental
    absolute = np.abs(signed)
    return {
        'record_count': len(records),
        'mean_predicted_delta_g_kcal_mol': float(np.mean(predicted)),
        'mean_experimental_delta_g_kcal_mol': float(np.mean(experimental)),
        'mean_signed_error_kcal_mol': float(np.mean(signed)),
        'mean_absolute_error_kcal_mol': float(np.mean(absolute)),
        'root_mean_square_error_kcal_mol': float(np.sqrt(np.mean(np.square(signed)))),
        'maximum_absolute_error_kcal_mol': float(np.max(absolute)),
        'ge_1_0_count': int(np.count_nonzero(absolute >= 1.0)),
        'ge_1_0_fraction': float(np.mean(absolute >= 1.0)),
        'ge_1_5_count': int(np.count_nonzero(absolute >= 1.5)),
        'ge_1_5_fraction': float(np.mean(absolute >= 1.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    prereg_sha256 = _sha256(PREREG_PATH)
    runner_sha256 = _sha256(RUNNER_PATH)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index in range(EXPECTED_RECORD_COUNT):
        path = args.input_dir / f'index-{index:03d}.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text())
        if row.get('selection_index') != index:
            raise ValueError(f'Selection index drift in {path}.')
        if row.get('preregistration_sha256') != prereg_sha256:
            raise ValueError(f'Preregistration drift in {path}.')
        if row.get('runner_sha256') != runner_sha256:
            raise ValueError(f'Runner drift in {path}.')
        if row.get('partition') != 'development':
            raise ValueError(f'Partition drift in {path}.')
        if row.get('confirmation_partition_opened') is not False:
            raise ValueError(f'Confirmation-open flag drift in {path}.')
        if row.get('do_not_commit') is not True:
            raise ValueError(f'Do-not-commit flag drift in {path}.')
        if row.get('status') == 'pass':
            records.append(row)
        else:
            failures.append(row)

    aggregate = _metrics(records) if records else None
    per_solvent_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        per_solvent_rows[row['canonical_solvent']].append(row)
    per_solvent = {
        solvent: _metrics(rows) for solvent, rows in sorted(per_solvent_rows.items())
    }
    threshold = 1.5
    target_passed = (
        not failures
        and aggregate is not None
        and aggregate['record_count'] == EXPECTED_RECORD_COUNT
        and aggregate['mean_absolute_error_kcal_mol'] <= threshold
    )
    payload = {
        'artifact': 'route2-hybrid-smd-development-full-verification-v2',
        'status': 'pass' if target_passed else 'fail',
        'do_not_commit': True,
        'partition': 'development',
        'confirmation_partition_opened': False,
        'record_count': EXPECTED_RECORD_COUNT,
        'success_count': len(records),
        'failure_count': len(failures),
        'failed_selection_indices': [row['selection_index'] for row in failures],
        'preregistration_sha256': prereg_sha256,
        'runner_sha256': runner_sha256,
        'aggregator_sha256': _sha256(Path(__file__)),
        'hard_accuracy_target': {
            'metric': 'mean_absolute_error_kcal_mol',
            'comparison': '<=',
            'threshold_kcal_mol': threshold,
            'passed': target_passed,
        },
        'aggregate_metrics': aggregate,
        'per_solvent_metrics': per_solvent,
        'claim_boundary': (
            'Complete frozen MNSol development-only hybrid full-solvation energy '
            'evaluation; no fitting/calibration, sealed confirmation, force/virial/'
            'Hessian accuracy, or production admission.'
        ),
    }
    payload['verification_sha256'] = _canonical_sha256(payload)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if target_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

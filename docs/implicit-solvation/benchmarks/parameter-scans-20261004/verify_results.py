"""Validate the portable result package and reaggregate FreeSolv with stdlib only."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_CONFIGS = {
    'fp64-lmax-high': {(1202, l) for l in (7, 9, 11, 13, 15)},
    'fp64-lmax-low': {(1202, l) for l in (1, 2, 3, 4, 5)},
    'fp64-grid': {(g, 15) for g in (110, 302, 590, 770, 974, 1202)},
    'fp32-historical-comparison': {(1202, 15)},
    'fp32-original-dense': {(g, l) for g in (194, 302) for l in range(1, 8)},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isfinite(float(a)) and math.isfinite(float(b)), message)
    require(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=5e-12), message)


def read_csv(name):
    with (ROOT / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def run():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    require(manifest['schema'] == 'parameter-scan-public-results-manifest-v1', 'manifest schema')
    expected = set(manifest['files']) | {'manifest.json'}
    require({p.name for p in ROOT.iterdir() if p.is_file()} == expected, 'file set')
    for name, receipt in manifest['files'].items():
        require(Path(name).name == name and not (ROOT / name).is_symlink(), 'unsafe file name')
        content = (ROOT / name).read_bytes()
        require(len(content) == receipt['size_bytes'], 'file size: ' + name)
        require(hashlib.sha256(content).hexdigest() == receipt['sha256'], 'file hash: ' + name)
        if name.endswith(('.json', '.csv')):
            require(not any(x in content for x in (b'mnsol-entry-', b'mnsol:0', b'/home/', b'positions_angstrom', b'source_values')), 'private payload: ' + name)
    summaries = read_csv('configuration-summary.csv')
    rows = read_csv('freesolv-records.csv')
    require(len(summaries) == 83 and len(rows) == 19902, 'coverage')
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        require(row['dataset'] == 'freesolv', 'row-level dataset boundary')
        key = (row['study'], int(row['n_lebedev']), int(row['lmax']))
        require(row['study'] in EXPECTED_CONFIGS and key[1:] in EXPECTED_CONFIGS[key[0]], 'unexpected configuration')
        unique = key + (row['index'],)
        require(unique not in identities, 'duplicate row')
        identities.add(unique)
        require(row['record_id'].startswith('mobley_'), 'FreeSolv identity')
        require(row['raw_finite'] in ('True', 'False'), 'finite flag')
        groups[key].append(row)
    require(len(groups) == 31, 'FreeSolv configuration count')
    summary_keys = set()
    for summary in summaries:
        study, dataset = summary['study'], summary['dataset']
        grid, lmax = int(summary['n_lebedev']), int(summary['lmax'])
        key = (study, dataset, grid, lmax)
        require(key not in summary_keys, 'duplicate summary')
        summary_keys.add(key)
        require(study in EXPECTED_CONFIGS and (grid, lmax) in EXPECTED_CONFIGS[study], 'summary configuration')
        require(dataset in ('freesolv', 'mnsol', 'combined'), 'summary dataset')
        n, finite = int(summary['expected_records']), int(summary['finite_records'])
        require(0 <= finite <= n and n - finite == int(summary['failed_or_blocked_records']), 'failure accounting')
        if study.startswith('fp32'):
            require(summary['full_panel_mae_kcal_mol'] == '' and summary['common_gate_pass_count'] == '0', 'FP32 admission boundary')
        if summary['mae_kcal_mol']:
            close(float(summary['mae_kcal_mol']) - float(summary['reference_mae_kcal_mol']), summary['mae_change_kcal_mol'], 'MAE change')
        if dataset != 'freesolv':
            continue
        members = groups[(study, grid, lmax)]
        require(len(members) == n == 642, 'FreeSolv row count')
        require({int(r['index']) for r in members} == set(range(1, 643)), 'FreeSolv index set')
        selected = [r for r in members if r['raw_finite'] == 'True']
        require(len(selected) == finite, 'finite count')
        for row in members:
            if row['raw_finite'] == 'False':
                require(all(row[k] == '' for k in ('prediction_kcal_mol', 'signed_error_kcal_mol', 'delta_vs_reference_kcal_mol')), 'failure exposes values')
            else:
                close(float(row['prediction_kcal_mol']) - float(row['reference_prediction_kcal_mol']), row['delta_vs_reference_kcal_mol'], 'paired delta')
        if not summary['mae_kcal_mol']:
            require(summary['metric_panel'] == 'unavailable-full' and finite < n, 'unavailable full MAE')
            continue
        require(int(summary['metric_panel_n']) == finite, 'metric panel count')
        errors = [float(r['signed_error_kcal_mol']) for r in selected]
        deltas = [float(r['delta_vs_reference_kcal_mol']) for r in selected]
        reference_errors = [e - d for e, d in zip(errors, deltas)]
        for value, field in ((statistics.fmean(map(abs, errors)), 'mae_kcal_mol'), (math.sqrt(statistics.fmean(e*e for e in errors)), 'rmse_kcal_mol'), (statistics.fmean(map(abs, reference_errors)), 'reference_mae_kcal_mol'), (statistics.fmean(map(abs, deltas)), 'mean_abs_energy_delta_kcal_mol'), (max(map(abs, deltas)), 'max_abs_energy_delta_kcal_mol')):
            close(value, summary[field], 'FreeSolv aggregate: ' + field)
    require(Counter(r['study'] for r in summaries) == {'fp64-lmax-high': 10, 'fp64-lmax-low': 10, 'fp64-grid': 18, 'fp32-historical-comparison': 3, 'fp32-original-dense': 42}, 'study aggregate counts')
    print(json.dumps({'status': 'PASS', 'configuration_summaries': 83, 'freesolv_rows': 19902, 'freesolv_configurations_reaggregated': 31, 'mnsol_scope': 'aggregate-only; not row-level reverified', 'scientific_solves_run': 0}, sort_keys=True))


if __name__ == '__main__':
    run()

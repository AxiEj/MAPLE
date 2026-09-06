from pathlib import Path
import hashlib, json, math, os, subprocess, sys, time

BASE = Path(__file__).resolve().parent
LOCK_PATH = BASE / 'preregistered-panel.json'
LOCK_BYTES = LOCK_PATH.read_bytes()
LOCK = json.loads(LOCK_BYTES)
SNAPSHOT = BASE / 'source-snapshot'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def verify_identity():
    assert LOCK_PATH.read_bytes() == LOCK_BYTES, 'Panel lock drift'
    assert sha(Path(LOCK['checkpoint'])) == LOCK['checkpoint_sha256'], 'Checkpoint drift'
    for rel, expected in LOCK['source_files_sha256'].items():
        assert sha(SNAPSHOT / rel) == expected, f'Source drift: {rel}'

def save(rows):
    good = [r for r in rows if r['status'] == 'ok']
    errors = [r['error_kcal_mol'] for r in good]
    metrics = None if not errors else {
        'n': len(errors), 'mae_kcal_mol': sum(map(abs, errors)) / len(errors),
        'rmse_kcal_mol': math.sqrt(sum(e * e for e in errors) / len(errors)),
        'mean_signed_error_kcal_mol': sum(errors) / len(errors),
        'maximum_absolute_error_kcal_mol': max(map(abs, errors)),
    }
    report = {'panel_id': LOCK['id'], 'lock_sha256': hashlib.sha256(LOCK_BYTES).hexdigest(),
              'n_selected': len(LOCK['records']), 'n_attempted': len(rows),
              'n_success': len(good), 'n_failed': len(rows) - len(good),
              'complete': len(rows) == len(LOCK['records']),
              'metrics_scope': 'entire_locked_small5' if len(good) == len(LOCK['records']) else 'successful_subset_only',
              'metrics': metrics, 'records': rows, 'scientifically_valid': False}
    tmp = BASE / 'results.json.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(BASE / 'results.json')

if __name__ == '__main__':
    verify_identity()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(SNAPSHOT), PYTHONNOUSERSITE='1')
    origin = subprocess.check_output([sys.executable, '-c', 'import maple; print(maple.__file__)'], cwd=SNAPSHOT, env=env, text=True).strip()
    assert Path(origin).resolve() == SNAPSHOT / 'maple/__init__.py', origin
    (BASE / 'runtime-origin.txt').write_text(origin + '\n')
    rows = []
    for record in LOCK['records']:
        verify_identity()
        cid = record['compound_id']; job = BASE / 'jobs' / cid
        assert sha(job / 'job.inp') == LOCK['input_sha256'][cid]
        assert sha(job / 'geometry.mol2') == record['mol2_sha256']
        if (job / 'console.log').exists():
            raise FileExistsError(f'Refusing to overwrite run: {job}')
        row = {k: record[k] for k in ['compound_id', 'name', 'natoms', 'experimental_kcal_mol', 'experimental_uncertainty_kcal_mol', 'experimental_reference']}
        start = time.monotonic()
        with (job / 'console.log').open('w') as stream:
            try:
                proc = subprocess.run([sys.executable, '-m', 'maple.main', str(job / 'job.inp')], cwd=SNAPSHOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=LOCK['timeout_seconds_per_record'])
                row['returncode'] = proc.returncode
                row['status'] = 'ok' if proc.returncode == 0 else 'execution_failed'
            except subprocess.TimeoutExpired:
                row.update(status='timeout', returncode=None)
        row['seconds'] = round(time.monotonic() - start, 3)
        if row['status'] == 'ok':
            try:
                path = job / 'job.out.implicit/mace-polar-ef-smooth-pcm-result.json'
                result = json.loads(path.read_text())
                assert result['profile'] == LOCK['profile']
                assert result['scientifically_valid'] is False
                components = result['components_hartree']
                delta = components['delta_g_solv']
                assert math.isclose(delta, components['electrostatic'] + components['cds'], abs_tol=1e-12)
                prediction = delta * LOCK['hartree_to_kcal_mol']
                assert math.isfinite(prediction)
                row.update(predicted_kcal_mol=prediction,
                           error_kcal_mol=prediction - record['experimental_kcal_mol'],
                           absolute_error_kcal_mol=abs(prediction - record['experimental_kcal_mol']),
                           components_kcal_mol={k: v * LOCK['hartree_to_kcal_mol'] for k, v in components.items()},
                           passivity_passed=result['electronic_passivity']['passivity_passed'],
                           scf_iterations=result['iterations'],
                           maximum_source_residual=result['maximum_source_residual'],
                           raw_result_sha256=sha(path), output_sha256=sha(job / 'job.out'))
            except Exception as exc:
                row.update(status='artifact_validation_failed', error=f'{type(exc).__name__}: {exc}')
        if row['status'] != 'ok':
            row['console_tail'] = (job / 'console.log').read_text()[-6000:]
        rows.append(row); save(rows)
        print(json.dumps(row, allow_nan=False), flush=True)
    verify_identity()

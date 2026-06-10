# PBC-MD release checklist

The MD-validation CI workflow was removed (single-developer fork; the ~10 min/push
Actions run was overhead without a matching benefit). The release gate is therefore
this manual checklist / command sequence, documented here. Run the backend-free
engine gate in §1 before tagging or proposing `fix/pbc` upstream; a production
claim for real PBC backends also requires the per-target reports and aggregate
checker in §2.

Allowed approval wording before §1 and §2 are complete:

> Static code review passed; `fix/pbc` is acceptable as a release-candidate /
> production-validation branch. It is **not** production-validated until the
> current clean commit has non-smoke PASS artifacts for every required target
> backend and `scripts/check_production_backend_matrix.py` passes over that
> report bundle.

Do not write "production-ready", "production validated", or "safe to roll out"
from static review, `--quick`, LJ-only evidence, or stale artifacts.

## 1. Unit + acceptance (backend-free, required)

```bash
# Unit layer (fake / LJ-reference calculators; default markers, never skipped)
pytest tests/dispatcher/md tests/calculator tests/read -q

# Production acceptance matrix (LJ reference, 11 non-skippable classes) — the backend-free MD-engine/LJ-contract gate
python scripts/production_validation.py; echo "exit=$?"
```

`production_validation.py` exits 0 **only if** every acceptance class PASSes **and**
none is skipped (a skip is inconclusive, not a pass). A fired barostat clamp
(`barostat_clamp_free`) fails the matrix; an unknown neighbor cutoff also fails
for calculators in the single-image MIC scope. The dated JSON/markdown report
under `validation/reports/` records the thresholds version, git commit,
per-class status, the pass/fail/skip summary, the barostat clamp count, and the
calculator unit contract + cutoff policy.

Treat the backend-free engine gate as passed only when the report / exit code shows:
- exit code `0`
- summary `n_fail == 0` and `n_skip == 0`
- `barostat clamps == 0`
- units `Ha / Ha/A / eV/A^3`

The 11 non-skippable engine classes are:

- `nve_energy_drift`
- `restart_determinism`
- `nvt_mean_temperature`
- `npt_pressure`
- `npt_com_pressure_invariance`
- `npt_volume_fluctuation`
- `npt_effective_energy_drift`
- `barostat_clamp_free`
- `stress_finite_difference`
- `pbc_geometry`
- `constraints_rejected`

The unit layer must also keep the admission/failure contracts green: raw ASE
calculators are rejected unless wrapped by `wrap_ase_calculator(...)`; declared
single-image cutoff / minimum-image / rank-3 cell failures are hard errors; the
NPT runtime MIC guard fails after barostat shrinkage before the next force/stress
evaluation for calculators in the single-image MIC scope;
Berendsen is rejected by default; and c-rescale clamp continuation remains
experimental-only.

## 2. Real-backend validation (per production backend; needs weights / CUDA)

```bash
pytest tests/integration -m integration -q          # NVE / NVT / NPT / stress smoke per backend
```

Run `production_validation.py` once for every target listed in
`validation/required_pbc_backends.toml`, including every AIMNet2/AIMNet2-NSE
Coulomb mode (`dsf`, `ewald`, `pme`), every declared MACE-MP / MACEPol PBC size,
and the UMA `omat` task. The examples below are representative only; the TOML
file is the exhaustive target list, and the aggregate checker enforces it.

```bash
python scripts/production_validation.py --model aimnet2-pbc --device cuda --model-option coulomb=dsf --model-option cutoff=5.0
python scripts/production_validation.py --model aimnet2-pbc --device cuda --model-option coulomb=ewald
python scripts/production_validation.py --model aimnet2-pbc --device cuda --model-option coulomb=pme
python scripts/production_validation.py --model mace-mp-pbc-small --device cuda
python scripts/production_validation.py --model macepol-pbc-small --device cuda
python scripts/production_validation.py --model uma --device cuda --model-option task=omat --model-option size=uma-s-1p1
```

After all target reports exist, the aggregate checker is mandatory:

```bash
python scripts/check_production_backend_matrix.py
```

Each run's provenance manifest (`*_md_manifest.json`) must record:
- `calculator.capabilities`: `energy_unit=Ha`, `force_unit=Ha/A`, `stress_unit=eV/A^3`,
  a finite `neighbor_cutoff_A`, and when applicable separate
  `local_descriptor_cutoff_A`, `short_range_realspace_cutoff_A`, and
  `long_range_coulomb_cutoff_A` fields, plus the cutoff scope flags
  `requires_single_image_mic` and `periodic_neighborlist_multi_image_safe`;
- `run.unit_contract` and `run.cutoff_policy.allow_unknown_cutoff == false`;
- `run.velocity_state_policy`: `load_state=true` must either be recorded as
  unconditioned (initialization-only COM/angular removal does not subtract DOF)
  or `condition_loaded_velocities=true` must record explicit projection/rescale;
- for NPT: `run.barostat.mode == "isotropic"` and `run.barostat_clamps.count == 0`.

## 3. Scope reminders (do not over-claim)
- `barostat=berendsen` is **equilibration-only**. It suppresses volume
  fluctuations and is rejected by default; setting
  `allow_equilibration_only_barostat=true` opts into an experimental,
  non-production equilibration run.
- NPT is **isotropic hydrostatic only** — no shear, cell-shape, or surface-tension
  control; it is not a Parrinello-Rahman / MTTK barostat.
- The unwrapped XYZ sidecar / `get_unwrapped_positions()` is a **per-atom**
  image-flag reconstruction, not a molecule-whole unwrap. In variable-cell NPT it
  mixes continuous atom motion with affine cell strain; do not use it as a
  fixed-cell MSD/diffusion coordinate without additional cell-strain handling.
- Periodic restart/load-state requires RST image flags. Legacy PBC checkpoints
  without image counters are rejected rather than silently zero-filled because
  unwrapped continuity across the handoff is unknowable.
- A raw ASE calculator enters MD only through `wrap_ase_calculator(...)` (a real unit
  conversion that then declares the contract), never by attribute-stamping.
- The smoke thresholds (`validation/thresholds.smoke.toml`) are for the unit layer only;
  the ship gate uses the production profile `validation/thresholds.toml`.

## 4. PBC neighbor-cutoff policy (per backend)

MAPLE has two explicit PBC cutoff scopes:

1. **Single-image MIC / safe-supercell scope** (default for user-supplied and
   wrapped calculators): MAPLE gates PBC MD on
   `neighbor_cutoff_A < minimum_image_radius_A` (Allen & Tildesley 2017, §1.5).
   Equality is rejected deliberately — a box with `L = 2 * cutoff` places an atom
   exactly at its periodic image's interaction surface and float rounding (or
   barostat shrinkage) flips the comparison silently.
2. **Multi-image-safe periodic-backend scope**: official periodic backends may
   declare `maple_periodic_neighborlist_multi_image_safe=true` and
   `maple_requires_single_image_mic=false`. For those calculators, MAPLE records
   the effective cutoff for provenance but does not force primitive-cell runs
   into the single-image supercell gate.

Per-backend effective cutoff recorded in provenance:

| Backend                         | Effective neighbor cutoff                              | Notes |
|---------------------------------|--------------------------------------------------------|-------|
| `aimnet2-pbc` / `aimnet2nse-pbc` (DSF)   | `max(5.0, public_cutoff_A)` (default 15.0)    | DSF is finite real-space and remains single-image MIC scoped; the required validation target uses `cutoff=5.0` for the 13.2 Å validation cell |
| `aimnet2-pbc` / `aimnet2nse-pbc` (Ewald / PME) | `5.0` (AEV short range)                  | Ewald/PME ignore the public cutoff at runtime |
| `mace-mp-pbc-*`                 | `models[0].r_max` (typically ~6 Å)                     | Official MACE PBC declares multi-image-safe scope |
| `macepol-pbc-*`                 | `models[0].r_max`                                      | Same logic as MACE-MP |
| `uma`                           | `6.0` (FAIR Chemistry graph radius)                    | UMA declares multi-image-safe scope |

For real-backend release validation, copy each backend's reported
`calculator.capabilities.neighbor_cutoff_A` plus any backend-specific local /
real-space / long-range cutoff fields from its provenance manifest into the
release log. The MIC-gated neighbor cutoff must be finite and match the table;
for single-image MIC calculators, an unknown cutoff is rejected at the MD
admission gate and cannot reach the report.

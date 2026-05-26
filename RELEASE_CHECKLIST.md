# PBC-MD release checklist

The MD-validation CI workflow was removed (single-developer fork; the ~10 min/push
Actions run was overhead without a matching benefit). The release gate is therefore a
manual command, documented here. Run it before tagging a release or proposing
`fix/pbc` upstream.

## 1. Unit + acceptance (backend-free, required)

```bash
# Unit layer (fake / LJ-reference calculators; default markers, never skipped)
pytest tests/dispatcher/md tests/calculator tests/read -q

# Production acceptance matrix (LJ reference, 10 non-skippable classes) — the sole gate
python scripts/production_validation.py; echo "exit=$?"
```

`production_validation.py` exits 0 **only if** every acceptance class PASSes **and**
none is skipped (a skip is inconclusive, not a pass). A fired barostat clamp
(`barostat_clamp_free`) or an unknown neighbor cutoff (the MD admission gate) already
fails the matrix. The dated JSON/markdown report under `validation/reports/` records the
thresholds version, git commit, per-class status, the pass/fail/skip summary, the
barostat clamp count, and the calculator unit contract + cutoff policy.

Ship only when the report / exit code shows:
- exit code `0`
- summary `n_fail == 0` and `n_skip == 0`
- `barostat clamps == 0`
- units `Ha / Ha/A / eV/A^3`

## 2. Real-backend validation (per production backend; needs weights / CUDA)

```bash
pytest tests/integration -m integration -q          # NVE / NVT / NPT / stress smoke per backend
python scripts/production_validation.py --model aimnet2-pbc      --device cuda
python scripts/production_validation.py --model mace-mp-pbc-small --device cuda
python scripts/production_validation.py --model macepol-pbc-small --device cuda
python scripts/production_validation.py --model uma               --device cuda
```

Each run's provenance manifest (`*_md_manifest.json`) must record:
- `calculator.capabilities`: `energy_unit=Ha`, `force_unit=Ha/A`, `stress_unit=eV/A^3`,
  a finite `neighbor_cutoff_A`, and when applicable separate
  `local_descriptor_cutoff_A`, `short_range_realspace_cutoff_A`, and
  `long_range_coulomb_cutoff_A` fields;
- `run.unit_contract` and `run.cutoff_policy.allow_unknown_cutoff == false`;
- for NPT: `run.barostat.mode == "isotropic"` and `run.barostat_clamps.count == 0`.

## 3. Scope reminders (do not over-claim)
- NPT is **isotropic hydrostatic only** — no shear, cell-shape, or surface-tension
  control; it is not a Parrinello-Rahman / MTTK barostat.
- A raw ASE calculator enters MD only through `wrap_ase_calculator(...)` (a real unit
  conversion that then declares the contract), never by attribute-stamping.
- The smoke thresholds (`validation/thresholds.smoke.toml`) are for the unit layer only;
  the ship gate uses the production profile `validation/thresholds.toml`.

## 4. PBC neighbor-cutoff policy (per backend)

MAPLE gates PBC MD on `neighbor_cutoff_A < minimum_image_radius_A` (Allen &
Tildesley 2017, §1.5). Equality is rejected deliberately — a box with
`L = 2 * cutoff` places an atom exactly at its periodic image's interaction
surface and float rounding (or barostat shrinkage) flips the comparison
silently. Enlarge the cell or reduce the cutoff; the gate is not relaxed.

Per-backend effective cutoff used by the gate:

| Backend                         | Effective neighbor cutoff                              | Notes |
|---------------------------------|--------------------------------------------------------|-------|
| `aimnet2-pbc` / `aimnet2nse-pbc` (DSF)   | `max(5.0, public_cutoff_A)` (default 15.0)    | DSF's public cutoff participates; AEV short range (5 Å) is the floor |
| `aimnet2-pbc` / `aimnet2nse-pbc` (Ewald / PME) | `5.0` (AEV short range)                  | Ewald/PME ignore the public cutoff at runtime |
| `mace-mp-pbc-*`                 | `models[0].r_max` (typically ~6 Å)                     | Read from the loaded MACE committee, not hard-coded |
| `macepol-pbc-*`                 | `models[0].r_max`                                      | Same logic as MACE-MP |
| `uma`                           | `6.0` (FAIR Chemistry graph radius)                    | Fixed: FAIRChem's `AtomicData.from_ase(radius=6.0)` is the only neighbor list UMA reads |

For real-backend release validation, copy each backend's reported
`calculator.capabilities.neighbor_cutoff_A` plus any backend-specific local /
real-space / long-range cutoff fields from its provenance manifest into the
release log. The MIC-gated neighbor cutoff must be finite and match the table;
an unknown cutoff is rejected at the MD admission gate and cannot reach the report.

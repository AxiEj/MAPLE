# PBC-MD production-hardening — release notes

## Conclusion first: release-candidate boundary

Do **not** treat `fix/pbc` as already production-validated or ready for direct
rollout. The code-level review, algorithm check, and comparison against the
literature / mature MD software did **not** find a blocking MD/PBC core defect
that would by itself fabricate dynamics or corrupt PBC trajectories. That is a
release-candidate finding, not a production claim.

The implementation is not a patch pile: MD concerns that were previously mixed in
`utils.py` are now separated into focused units / PBC / DOF / pressure / velocity
initialization / motion projection / capabilities / semantics / provenance /
validation modules, while `utils.py` remains a compatibility re-export shim with
the module responsibilities documented explicitly.

Review boundary: this statement is based on static code review plus
literature/documentation cross-checks. It does **not** say that the full
acceptance matrix was run in the current environment, and it does **not** replace
real-backend production report artifacts. The correct release statement is:
static review passed; `fix/pbc` may enter controlled production validation; final
production enablement must cite same clean commit, target backend, target hardware
non-smoke PASS reports plus the aggregate backend-matrix checker.

| Area | Review conclusion / boundary |
|------|------------------------------|
| Fixed-cell PBC + NVE/NVT | Aligned with standard MD design; no core theory error found. |
| Image flags / wrapped-unwrapped coordinates | Correct per-atom image-counter direction, restartable, triclinic-aware; not a molecule-whole unwrap. |
| Velocity-Verlet | Standard kick-drift-wrap-force-kick shape; PBC wrapping occurs after drift, not only at output. |
| V-rescale thermostat | Consistent with Bussi-Donadio-Parrinello / GROMACS-style stochastic velocity rescaling. |
| Langevin-middle | Uses LF-middle / BAOAB-style half-step velocity semantics and records the velocity representation. |
| NPT c-rescale | Acceptable as isotropic hydrostatic stochastic cell rescaling; not Parrinello-Rahman / MTTK, shear, surface-tension, or anisotropic cell-shape sampling. |
| Berendsen barostat | Correctly scoped as equilibration-only and rejected by default for production-style NPT. |
| Pressure / stress | ASE Voigt stress sign, kinetic/configurational pressure split, and DOF-policy COM kinetic exclusion are handled explicitly. |
| PBC admission gates | Unit, PBC support, single-image cutoff/MIC scope, multi-image-safe backend scope, full-rank cell, stress, and constraint gates fail closed instead of fabricating plausible output. |
| Restart / trajectory / provenance | Image flags, full cell matrix, velocity representation, RNG/provenance, cutoff policy, and barostat mode are persisted or checked. |
| Validation | Acceptance design is suitable, but production status requires actually running the non-smoke matrix for every required target. |

Non-blocking follow-ups from the review:

- Avoid unnecessary `Atoms.info` side effects in the UMA adapter when setting
  default spin / charge metadata.
- Keep AIMNet2 Ewald/PME claims tied to official backend behavior plus stress-FD
  and acceptance artifacts; the MIC gate only proves the local descriptor cutoff.
- Treat the NPT volume-fluctuation check as a loose sanity class, not a high-
  precision ensemble proof by itself.

External alignment anchors used for the review:

- LAMMPS wrapped/unwrapped coordinates and image flags:
  <https://docs.lammps.org/dump.html>
- Bussi-Donadio-Parrinello stochastic velocity rescaling:
  <https://arxiv.org/abs/0803.4060>
- GROMACS thermostat / barostat guidance:
  <https://manual.gromacs.org/current/reference-manual/algorithms/molecular-dynamics.html>
- OpenMM `LangevinMiddleIntegrator` half-step / BAOAB semantics:
  <https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html>
- Bernetti-Bussi stochastic cell rescaling:
  <https://arxiv.org/abs/2006.09250>
- ASE `Atoms.get_stress(voigt=True, include_ideal_gas=False)` semantics:
  <https://ase-lib.org/ase/atoms.html>

This pass makes the PBC molecular-dynamics path production-honest: correct where
supported, hard-reject where not. The user-visible behavior changes below are
grouped by the workstream that introduced them.

## MD default parameters now come solely from the ensemble dataclasses (WS1)

`CommandControl.DEFAULTS["md"]` previously duplicated every physics default and
silently overrode the `NVEParams` / `NVTParams` / `NPTParams` dataclasses. It now
contains only `{"ensemble": "nve"}`; all other defaults are resolved from the
dataclasses at construction time. Precedence is **explicit inline > MDP file >
dataclass default**, and no physics default is duplicated in two places.

Concrete changes to a *bare* run (e.g. `#md`, `#md(ensemble=nvt)`), all of which
previously used the hard-coded `DEFAULTS["md"]` values:

| Parameter           | Old (DEFAULTS) | New (dataclass) | Notes |
|---------------------|----------------|-----------------|-------|
| `timestep`          | 0.25 fs        | 0.1 fs          | all ensembles |
| `steps`             | 400000         | 100000          | all ensembles (= 10 ps at 0.1 fs) |
| NVE `remove_com_every` | 100         | 0               | strict NVE: no runtime COM projection by default, so diffusion/VACF transport observables are not perturbed |
| NVE `remove_angular`   | False       | True            | isolated-molecule init draw projects out rigid-body rotation once |
| NPT `thermostat`    | langevin       | v-rescale       | the recommended canonical production thermostat (Bussi 2007) |
| NPT `tau_t`         | 100 fs         | 200 fs          | gentler coupling when thermostat + barostat both act each step |

Other consequences:

- `#md(verbose=N)` and `#md(allow_partial_pbc=...)` are now accepted — the `#md`
  whitelist is the full union of dataclass fields, not the old `DEFAULTS["md"]`
  key set (which omitted `verbose`).
- `CommandControl.summary()` for a bare run now lists only the keys the user
  actually set; the authoritative resolved-parameter dump is the ensemble's
  own `*** MD PARAMETERS ***` block (`_log_parameters()`).

## DOF / temperature self-consistency (WS0-A, ships with WS1)

The initial Maxwell–Boltzmann velocity draw is now rescaled to the **init**
degree-of-freedom basis (the projection actually applied at initialization),
while runtime temperature, the thermostat target, and the summary use an
**operator-aware runtime** basis:

- A mode projected at init is subtracted from the runtime DOF only if the per-step
  operator cannot repopulate it. NVE (deterministic), v-rescale and c-rescale
  (global scalars) cannot; Langevin (per-atom noise) can.
- This fixes a latent over-heating: with the new NVE defaults an isolated molecule
  projects COM + rotation (3N − 6 active DOF), but the old code rescaled the draw
  to the 3N runtime count, over-heating the internal modes by 3N/(3N − 6) (3× for
  water).
- The reported **initial temperature** is shown in the basis it was generated in.
  For Langevin a second line reports the runtime basis, which reads (init/runtime)
  × T at t = 0 because the thermostat has not yet repopulated the projected modes
  — a basis difference, not an error.
- **Input velocities now have explicit semantics.** If a fresh run has
  `atoms.arrays["velocities"]` and `init_velocities=true`, MAPLE treats those
  velocities as the initialization state: it applies the same COM/angular
  projection policy as a random Maxwell-Boltzmann draw and rescales to the
  init-DOF target temperature. If `init_velocities=false`, MAPLE consumes input
  velocities as provided and the DOF policy no longer silently assumes the
  initialization-only projection was applied; runtime projection still requires
  `remove_com_every` / `remove_angular_every`.

## NPT logging thermodynamic consistency (WS2)

The NPT thermo row is now internally consistent: `Press(bar)` / `Vol(A^3)` are the
**post-barostat-rescale** primary state (paired with the post-rescale T/KE/PE), and
the pre-rescale pair that drove the barostat decision is kept as labeled
`Press_pre(bar)` / `Vol_pre(A^3)` diagnostic columns. The summary mean pressure
uses the post-rescale series. For Langevin NPT the post-rescale pressure kinetic
term uses the synchronized standard velocity.

## NPT COM kinetic pressure closure + backend cutoff scope (WS11)

- **NPT pressure now follows the resolved COM DOF policy.** When COM translation
  is projected/constrained by the operator-aware DOF policy (the default
  v-rescale/c-rescale fresh PBC path), the kinetic pressure term uses
  `K - K_cm`; unconditioned input/load-state velocities keep COM active and use
  full `K`. The same flag is threaded through post-rescale logging and both
  c-rescale / Berendsen barostat pressure decisions. The acceptance matrix adds
  `npt_com_pressure_invariance`, an imposed-COM white-box gate proving a pure
  COM boost leaves active pressure and deterministic c-rescale volume response
  unchanged while the full-kinetic pressure would shift.
- **Cutoff/MIC support is explicit about scope.** User/wrapped calculators remain
  in the conservative single-image MIC-safe supercell scope and must satisfy
  `neighbor_cutoff_A < minimum_image_radius_A`. Official periodic backends can
  declare `maple_periodic_neighborlist_multi_image_safe=true` and
  `maple_requires_single_image_mic=false`, so MAPLE records their effective
  cutoff without rejecting valid primitive-cell periodic neighbor-list runs
  solely because the cutoff exceeds the Wigner-Seitz inradius. AIMNet2 DSF is
  the explicit exception: because it is a finite real-space cutoff mode, it
  remains MIC-scoped while AIMNet2 Ewald/PME use the multi-image-safe scope.
- **DCD PBC output has a triclinic round-trip regression.** The DCD test now
  writes/reads a triclinic PBC trajectory, checks frame count, `cellpar`,
  coordinate RMSD, and append header frame count.
- Thresholds bumped to **1.6.0** (the smoke profile mirrors this as
  `smoke-1.6.0`) so real-backend production reports must include the imposed-COM
  NPT pressure class.

## Hard rejects (WS0-B, WS0-C)

- **ASE constraints + MD → rejected.** The hand-written Velocity Verlet has no
  SHAKE/RATTLE/SETTLE, constraint-aware projection, or constraint DOF accounting,
  so constrained MD would silently fabricate kinetic energy. Constrained MD is a
  roadmap item; `opt/scan/ts/irc` are unaffected.
- **Partial periodicity + MD → rejected for production.** `allow_partial_pbc=true`
  opts into an EXPERIMENTAL (non-production) slab/partial-PBC trajectory and prints
  a banner; it does not pass production validation. `#pbc` still emits full 3-D PBC.

## Reversible c-rescale + effective-energy monitoring (WS7)

The stochastic cell-rescaling (c-rescale) barostat now uses the **reversible
λ = √V integrator** of Bernetti & Bussi (2020) — their Eq. 7, the "reversible
Euler integrator" of their Table I — instead of the simpler Euler scheme on the
log-volume ε. In the production `v-rescale + c-rescale` NPT driver this is now
ordered as the paper's reversible Euler scheme: propagate √V and scale
positions/momenta first, recompute forces at the scaled geometry, then perform a
full Velocity Verlet step and apply the stochastic velocity-rescale thermostat
to the full-step velocity. This is **not** the heavier symmetric Trotter
integrator. Propagating λ = √V makes the noise amplitude
`sqrt(k_B T β / 2τ_P)` *constant* (V-independent), removing the multiplicative-
noise discretization bias of the ε form and adding the exact Itô correction
`−k_B T/(2V)` to the drift (derived analytically from the ε-form SDE). Per-step
the cell scales by `μ = (λ_new/λ)^{2/3}` and velocities by `1/μ`; λ is re-derived
from the actual volume each step so the stability clamp cannot make the strain
variable drift from log V. Isotropic-only scope is unchanged (no shear / cell
shape; not a Parrinello–Rahman / MTTK replacement).

- **NPT conserved quantity is now reported (v-rescale + c-rescale).** The driver
  accumulates the thermostat, barostat and runtime-projection work into one
  external-work ledger, so the conserved quantity
  `H̃ = KE + PE + P_0·V − Σ ΔW_ext` is logged as the `H_cons(Ha)` thermo column
  (appended last, so existing column indices are unchanged) and its drift is the
  summary "H̃ drift". This closes the gap noted previously in the NPT loop, where
  the Bussi conserved quantity was not reported because the barostat work was not
  accumulated. For the LJ reference, H̃ holds to ~5e-7 Ha while the bare energy
  swings ~0.08 Ha. Langevin NPT (no conserved energy) and Berendsen
  (equilibration-only) do not report H̃.

## Restart/load-state velocity and image-flag safety (WS10)

- **`load_state=true` no longer assumes initialization projection.** Loaded RST
  velocities are treated as an unconditioned new-run state by default, so
  initialization-only `remove_com` / `remove_angular` do not silently subtract
  DOF unless the user explicitly sets `condition_loaded_velocities=true`. That
  opt-in converts any stored velocity representation to standard velocities,
  applies the same COM/angular projection and temperature rescale as a fresh
  initialization, and records the policy in the run manifest.
- **PBC legacy RST files without image flags are rejected.** Current RST files
  carry per-atom image counters. If a periodic checkpoint lacks those counters,
  MAPLE now hard-fails restart/load-state instead of zero-filling them, because
  unwrapped trajectory continuity across the handoff cannot be reconstructed.
- **The NPT volume-fluctuation acceptance gate exercises `barostat_stride=N_P`.**
  The production threshold profile now validates a strided c-rescale
  distribution-level run (`N_P>1`) in addition to the H̃ drift and zero-clamp
  gates, keeping the stride path out of the "white-box only" category.

## Release acceptance matrix: NVT gate fix + effective-energy class (WS7)

- **`nvt_mean_temperature` gate corrected.** The window is now `k_sigma` × the
  standard error of the **mean** temperature (block-averaged, with the i.i.d.
  `sqrt(n)` value as a floor), not `k_sigma` × the instantaneous spread. The
  previous form multiplied the standard error back by `sqrt(n)`, widening the
  window ~`sqrt(n)`× into a near-instantaneous-fluctuation band that could not
  catch a real mean-temperature bias. The run length, equilibration discard
  (0.4) and block count (10) are calibrated against the LJ reference so a correct
  thermostat stays < 4σ over many seeds while the ~2 K standard error keeps the
  ~12 K window sensitive to a real >12 K bias.
- **New `npt_effective_energy_drift` acceptance class.** Bounds the reversible
  c-rescale H̃ drift as a per-atom, per-ps slope (the NPT analogue of the NVE
  energy-drift check); pre-registered at 1e-6 Ha/atom/ps (LJ reference passes at
  ~4e-10, ~2000× margin; broken/absent work accounting drifts ~1.2e-5 and fails).
- Thresholds bumped to **1.2.0** (gate change + new class are versioned with the
  report, never tuned to results).

## Trajectory analysis caveat (WS7)

`get_unwrapped_positions` (and the unwrapped XYZ sidecar) is documented as a
**per-atom** image reconstruction using the **current** cell: it is not a
molecule-whole unwrap, and for a variable-cell (NPT) run it is not a fixed-cell
lab-frame coordinate, so variable-cell MSD/diffusion must account for the cell
strain separately. Fixed-cell (NVE/NVT) unwrapped coordinates are lab-frame and
suitable for MSD/VACF.

When `traj_format=dcd`, the DCD trajectory is a wrapped visualization trajectory:
it stores wrapped coordinates and the cell, but not MAPLE image flags. For
transport/MSD/continuous-coordinate analysis use the unwrapped XYZ sidecar or
reconstruct from RST/image flags, not the DCD alone.

## AIMNet2 long-range Coulomb admission boundary

For `aimnet2-pbc` / `aimnet2nse-pbc`, MAPLE can fully MIC-gate the DSF
real-space cutoff. For Ewald/PME modes the official AIMNet2 backend owns the
long-range real-space/reciprocal-space parameterization; MAPLE records those
settings in provenance and gates only the fixed 5 Å local AEV descriptor cutoff.
Production validation reports must therefore include each Coulomb mode they
intend to claim, not just a DSF or local-cutoff smoke.

## Real-backend production matrix is now an aggregate gate

`scripts/production_validation.py` validates one calculator target at a time.
Before MAPLE can claim production PBC-MD for real ML backends, the report bundle
must also pass:

```bash
python scripts/check_production_backend_matrix.py
```

The required matrix lives in `validation/required_pbc_backends.toml` and covers
`aimnet2-pbc` / `aimnet2nse-pbc` in DSF, Ewald and PME modes, every
`mace-mp-pbc-*` and `macepol-pbc-*` size, and the UMA `omat` PBC task. The
checker rejects LJ-only evidence, quick-smoke reports, dirty or old-commit
reports, skipped acceptance classes, c-rescale clamp events, missing manifests,
missing calculator unit/cutoff contracts, and any stress report that is not a
full ASE-Voigt `[xx, yy, zz, yz, xz, xy]` finite-difference pass with matching
signs. This makes the previous review requirement machine-checkable: LJ proves
the MD engine; real backend reports prove the production backend claim.

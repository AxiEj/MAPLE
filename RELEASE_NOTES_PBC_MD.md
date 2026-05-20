# PBC-MD production-hardening — release notes

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

## NPT logging thermodynamic consistency (WS2)

The NPT thermo row is now internally consistent: `Press(bar)` / `Vol(A^3)` are the
**post-barostat-rescale** primary state (paired with the post-rescale T/KE/PE), and
the pre-rescale pair that drove the barostat decision is kept as labeled
`Press_pre(bar)` / `Vol_pre(A^3)` diagnostic columns. The summary mean pressure
uses the post-rescale series. For Langevin NPT the post-rescale pressure kinetic
term uses the synchronized standard velocity.

## Hard rejects (WS0-B, WS0-C)

- **ASE constraints + MD → rejected.** The hand-written Velocity Verlet has no
  SHAKE/RATTLE/SETTLE, constraint-aware projection, or constraint DOF accounting,
  so constrained MD would silently fabricate kinetic energy. Constrained MD is a
  roadmap item; `opt/scan/ts/irc` are unaffected.
- **Partial periodicity + MD → rejected for production.** `allow_partial_pbc=true`
  opts into an EXPERIMENTAL (non-production) slab/partial-PBC trajectory and prints
  a banner; it does not pass production validation. `#pbc` still emits full 3-D PBC.

# Route 2 capability matrix

Capabilities are evidence-admitted, not inferred from a callable method or a
CLI restriction.

| tier | meaning |
| --- | --- |
| E | scalar energy |
| F | conservative force of that same scalar |
| H | force-derived numerical Hessian/HVP admitted for workflows |
| V | strict common variational electronic-continuum functional |
| M | public MD release gate |

## Conservative-vNext registry

| scalar/profile | E | F | H | V | M | current disposition |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| `route2-operational-cpcm-fixedtopology-electrostatic-v1` | no | no | no | no | no | scalar/state kernel implemented; legacy-width profile remains unadmitted |
| `route2-profile-operational-cpcm-fixedtopology-radialgto-electrostatic-v1` | no | no | no | no | no | real same-scalar derivative candidate; rotation/torque and physical-component gates failed |
| `route2-profile-diagnostic-fixedbox40-cpcm590-radialgto-electrostatic-v1` | no | no | no | no | no | earlier derivative/path panels pass, but the frozen all-panel water canary fails rotation energy and force covariance; retained as negative evidence, not admissible |
| `route2-profile-diagnostic-fixedbox{32,48,56}-cpcm590-radialgto-electrostatic-v1` | no | no | no | no | no | preregistered box controls passed at one equilibrium water geometry; distinct identities, no adaptive selection, no public capability |
| `route2-profile-diagnostic-fixedbox48-cpcm1202-radialgto-electrostatic-v1` | no | no | no | no | no | separately versioned higher-order candidate; same scalar and unchanged symmetry thresholds, no executed release evidence yet |
| `route2-profile-diagnostic-pairframe-cpcm110-radialgto-electrostatic-v1` | no | no | no | no | no | distinct ordered-pair-frame ensemble discretization; algebra/continuum tests pass and one preliminary unbound real methanol engineering run meets local thresholds, but clean source-bound PES/symmetry/accuracy evidence is absent |
| `route2-profile-diagnostic-ddx-ddpcm194-radialgto-electrostatic-v1` | no | no | no | no | no | full eight-channel joint `(psi,phi)` map is derived from one ddPCM scalar and passes local derivative tests; ddX finite-grid rotation drift and missing achieved algebraic-residual report keep it diagnostic |
| `route2-profile-diagnostic-cpcm-injectedgrid-radialgto-electrostatic-v1` | no | no | no | no | no | synthetic injected-grid diagnostic only |
| `route2-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1` | no | no | no | no | no | implemented diagnostic; exact-GTO mismatch prevents admission |
| `route2-operational-cpcm-fixedtopology-smdcds-v1` | no | no | no | no | no | blocked until electrostatic F/H and same-scalar CDS force pass |
| `route2-variational-common-functional-v1` | no | no | no | no | no | the current checkpoint's original intrinsic energy plus original four-channel source instantiation is formally ruled out by a source-bound real-checkpoint counterexample; the generic scalar identity remains disabled rather than being reassigned to a changed model |
| `route2-variational-macepolar-energygradient-fixedcavity-cpcm-v1` | no | no | no | no | no | scalar-first complete eight-channel effective-source candidate, fixed reciprocal C-PCM scalar, and their common constrained state/envelope kernel are implemented; a source-bound real-checkpoint water canary passes cold/warm replay and one three-step envelope FD, but the original density head remains diagnostic and passivity/root-uniqueness/combined-Hessian/structural-rotation/full-panel gates remain open |
| `route2-variational-macepolar-energygradient-fixedcavity-harmonicgalerkin-cpcm-v1` | no | no | no | no | no | complete-irrep coefficient action, exact-adjoint fixed-snapshot scalar, and common-state integration are implemented; the external coefficient snapshot is geometry independent and intentionally has zero continuum coordinate partial; every release gate remains missing |
| `route2-variational-macepolar-energygradient-smoothharmonicgalerkin-cpcm-v1` | no | no | no | no | no | distinct moving smooth weighted-overlap scalar; its Torch graph reassembles `E`, `K`, `V`, `A=E.T K E`, and `S=E.T V` and passes independent matrix parity, coordinate/mixed finite differences, and roundoff-level synthetic rotation covariance; the reference is `C1` but not generally `C2` at shell tangency; real-checkpoint common-state stability, physical calibration, and every release gate remain missing |

The first release target is the operational electrostatic profile. Tier V is
not required for it and must remain false unless the model energy/source
identity, reciprocity, stability, invertibility, and full coordinate derivative
are independently proven.

The decisive current negative artifact is
[`evidence/mace-conjugacy-nogo-d17c35ac/`](evidence/mace-conjugacy-nogo-d17c35ac/README.md).
At clean commit `d17c35ac`, both tested field states have a gauge-reduced
missing-radial witness relative magnitude above `0.731`, versus numerical-zero
thresholds near `5.23e-10`; both source/energy signs fail. This closes only the
"retain original energy and original source" route. The separately named
eight-channel energy-gradient source is a changed model identity and has no
admitted tier.

The corresponding positive-but-narrow changed-source implementation canary is
[`evidence/variational-common-water-576550e9/`](evidence/variational-common-water-576550e9/README.md).
It proves that one real water state of the new scalar converges and that its
stationary-envelope derivative matches three re-solved finite differences. It
does not override the `no` entries above: the sampled Lebedev continuum has no
structural global `SO(3)` guarantee, and the required passivity, uniqueness,
Hessian, domain, PES, and chemical gates are absent.

## Legacy baseline

At baseline `15777aad`, the old registry contains 20 experimental energy-only
profiles and one bounded experimental force profile. Those declarations do not
satisfy the new scalar/state/provenance and full-panel admission contract, so
their vNext tiers are all false. The exact per-profile snapshot is
`evidence/baseline-15777aad/capability-matrix.json`.

Separate paths remain fail-closed:

| path | vNext status |
| --- | --- |
| pyddx/ddX | same-scalar eight-channel diagnostic implemented; finite-grid rotation and achieved-residual gates remain failed/open |
| PCMSolver | independent energy/operator audit backend |
| source-dependent \(\rho\)-DROP | energy-only; coordinate VJP and Gates B/C missing |
| local-jet receiver | research diagnostic under a distinct identity |
| public MD | disabled until Tier M |

This file will be updated only when the corresponding evidence artifact is
source/model/runtime bound and passes every preregistered gate.

The callable water radial-GTO path is deliberately absent from public result
admission. At the current real-water canary it passed the three central
directional differences but missed rotation covariance (`2.515e-4` versus
`1e-4`) and torque (`1.078e-4 eV` versus `1e-4 eV`). It is therefore not a
conservative-force capability for MAPLE workflows.

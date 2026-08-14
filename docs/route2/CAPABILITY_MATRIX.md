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
| `route2-variational-common-functional-v1` | no | no | no | no | no | disabled pending every strict-variational gate |
| `route2-variational-macepolar-energygradient-fixedcavity-cpcm-v1` | no | no | no | no | no | scalar-first complete eight-channel effective-source candidate plus scalar-first fixed reciprocal C-PCM candidate; original density head is diagnostic only; combined stationary scalar/sign/gauge/passivity/root/envelope/rotation/release gates remain open |
| `route2-variational-macepolar-energygradient-fixedcavity-harmonicgalerkin-cpcm-v1` | no | no | no | no | no | complete-irrep coefficient action, immutable SPD snapshot, exact-adjoint stationary scalar, explicit coefficient-conjugation covariance, geometry-bound `C-infinity` harmonic overlap/multiplication, and the full eight-channel Gaussian `S/S.T` intertwiner are implemented; continuum Green translation assembly, analytic coordinate pullback, combined model-continuum stationarity, and every release gate remain missing |

The first release target is the operational electrostatic profile. Tier V is
not required for it and must remain false unless the model energy/source
identity, reciprocity, stability, invertibility, and full coordinate derivative
are independently proven.

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

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
| `route2-profile-diagnostic-fixedbox40-cpcm590-radialgto-electrostatic-v1` | no | no | no | no | no | one-water Cartesian/orientation, seven-geometry path/loop, equilibrium box-tail, and 20-molecule/60-geometry plus 11-path-point directional gates pass; full Cartesian panel, component physics, H/NVE, and admission remain open |
| `route2-profile-diagnostic-fixedbox{32,48,56}-cpcm590-radialgto-electrostatic-v1` | no | no | no | no | no | preregistered box controls passed at one equilibrium water geometry; distinct identities, no adaptive selection, no public capability |
| `route2-profile-diagnostic-cpcm-injectedgrid-radialgto-electrostatic-v1` | no | no | no | no | no | synthetic injected-grid diagnostic only |
| `route2-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1` | no | no | no | no | no | implemented diagnostic; exact-GTO mismatch prevents admission |
| `route2-operational-cpcm-fixedtopology-smdcds-v1` | no | no | no | no | no | blocked until electrostatic F/H and same-scalar CDS force pass |
| `route2-variational-common-functional-v1` | no | no | no | no | no | disabled pending every strict-variational gate |

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
| pyddx/ddX | energy-only until full PES smoothness/admission evidence |
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

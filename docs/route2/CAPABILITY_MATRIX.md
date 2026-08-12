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
| `route2-operational-cpcm-fixedtopology-electrostatic-v1` | no | no | no | no | no | target; not implemented/admitted |
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

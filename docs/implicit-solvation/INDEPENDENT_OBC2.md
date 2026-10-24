# Independent OBC-II in MAPLE

## Default and installation

Omitting `provider` for GB/OBC-II selects MAPLE's independent Torch provider.
It uses the frozen OBC-II/mbondi2 equations and supports `nonpolar=ace` or `none`.
The solvent runtime does not import OpenMM or query its installed metadata.

Install the CPU/CUDA PyTorch build appropriate for the machine first, then:

```bash
pip install '.[implicit-gb]'
```

The `implicit` and `implicit-obc2-torch` extras select the same Torch dependency.
Gas-model dependencies and checkpoints remain model-specific. Independence does
not eliminate input preparation: topology, atom types and fixed charges are
still required. Explicitly requesting AM1-BCC generation still requires its
charge-preparation tools; prepared MOL2 charges do not.

## Ordinary public input

```text
#model=ani2x(model_path=/path/to/ani2x.pt)
#sp
#device=cpu
#charge(source=mol2,label=fixed-input)
#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,experimental=true)

0 1
MOL2 /path/to/prepared-molecule.mol2
```

The normalized provider is `torch`; writing `provider=torch` explicitly gives
the same path. Solvent arithmetic is float64, on the requested model device.
Gas and solvent device/dtype are recorded separately. No failed independent
calculation switches to OpenMM, another nonpolar model or another dtype.

The common calculator adds the solvent energy/force exactly once to the gas
MLIP result. Charges, atom order, connectivity and parameter identity are frozen;
coordinates and effective Born radii remain live. The same scalar supplies
solvent forces, Hessians and direct HVPs. A direct HVP does not build a dense H.

## Analytic task admission

Provider capability alone does not qualify a composed MLIP task. Analytic
implicit FREQ/P-RFO/direct-HVP Dimer requests require a sealed exact-cell
qualification: ANI2x checkpoint
`c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa`, CPU,
float64, D4 disabled, OBC-II/mbondi2, and ACE or NONE.

The factory verifies the packaged receipt, current installed source inventory,
actual calculator implementation and model/solvent identities before enabling
both admission flags. Missing or stale qualification raises
`ANALYTIC_CELL_UNQUALIFIED`; there is no public input/environment bypass.

For a qualified installation, request `hessian=analytic,d4=false` in the model
header. Numerical Hessian and force-difference Dimer remain separately labeled
existing modes, not fallbacks for a rejected analytic request. Other MLIPs,
checkpoints, devices and dtypes do not inherit this analytic qualification.
IRC and broader implicit TS workflows are not added by this provider migration.

P-RFO also checks a fresh final Hessian before claiming saddle convergence;
small forces alone do not establish a transition state. A short P-RFO/Dimer
replay is execution evidence, not a converged chemical transition state.

## Explicit legacy OpenMM

```bash
pip install '.[implicit-openmm]'
```

Use `provider=openmm` for an OpenMM OBC-II calculation, LCPO, `inner=prebuilt`,
or OpenMM-specific `platform`, `precision` and device controls. For example:

```text
#solv(implicit=water,method=gb,provider=openmm,model=obc2,nonpolar=ace,platform=Reference,experimental=true)
```

Explicit HCT/OBC-I/GBn/GBn2 model selections retain their legacy OpenMM provider.
OpenMM-specific options never silently change an OBC-II provider selection.
The ddLPB and free-energy extras retain their previous dependencies and scope.
CHA remains a separate energy-only route; this change does not alter its
private exact-SP implementation or add derivatives.

## Evidence boundaries

Historical numerical reports are immutable. The new source-compatibility ledger
records source drift for audit only; it never authorizes reuse of an old source
freeze. Candidate qualification binds the complete checkout; runtime admission
checks wheel-visible code and data; installed activation separately binds the
final wheel and uses the installed `maple` console entrypoint without candidate
monkeypatches. No experimental solvation-accuracy or performance claim follows
from formula parity, derivative consistency or these task replays.

# Route 2 conservative-vNext user guide

## Current public status

There is currently **no admitted conservative-vNext Route-2 public profile**.
All ten registry profiles have `E=F=H=V=M=false`. The radial-GTO water path is
an internal validation candidate; it is deliberately unavailable through
MAPLE's public calculator/result API.

Its single-water equilibrium 32/40/48/56-Angstrom box audit passes the frozen
tail thresholds. Its preregistered 20-molecule directional PES panel also
passes, as do the 20-reference-geometry, 465-component Cartesian and separated
residual-refinement panels. None of these results admits energy or force for
public use because all-panel symmetry/loop, physical-component, multi-geometry
box, and public workflow gates are still open.

The stable public legacy Route-2 inputs remain experimental energy-only paths
with their historical contracts. They must not be interpreted as the vNext
same-scalar PES described here.

## Canonical vNext scalar (internal validation only)

The sole operational electrostatic scalar is

```text
E_op(R) = E_vac(R) + 0.5 * <c*(R), P_R(c*(R))>_Q
G_np    = 0
```

where `c*(R)` is the converged root of
`route2-constrained-mutual-polarization-root-v1`. Field-conditioned MACE energy
and SMD/CDS are not part of this scalar.

An internal result/evidence record must identify at least:

```text
profile_id
scalar_id
state_equation_id
checkpoint/provenance SHA256
continuum configuration contract
root hash and primal residual
adjoint residual (for derivative evidence)
topology hash
scalar leaves in eV
capability admission status
```

No caller may turn this evidence into a public result by setting a capability
flag manually; capability identity is registry-derived.

## Executing the real validation canaries

These are developer audits, not user calculations:

```bash
export MAPLE_ROUTE2_REAL_MACEPOL=1
export MAPLE_ROUTE2_MACE_DEVICE=cuda
export MAPLE_ROUTE2_METHANE_MOL2=/absolute/path/to/mobley_9055303.mol2
python -m pytest -q -s --disable-warnings \
  tests/route2_vnext/test_mace_polar_real_checkpoint.py \
  tests/route2_vnext/test_mace_polar_real_operational_scalar.py
```

The checkpoint and MOL2 bytes must match the evidence contract. Missing
advertised assets fail the real-stack job rather than producing a green skip.

## Unsupported examples (fail closed)

The following vNext requests are intentionally unsupported today:

```text
conservative-force single point
solution-phase OPT
NEB/CINEB, PRFO, Dimer, TS, IRC
numerical FREQ or HVP
MD
fixed-topology SMD-derived CDS
multi-solvent vNext profiles
rho-DROP forces
strict common variational MACE-continuum functional
```

They remain closed because no Tier-F/Tier-H/Tier-M release artifact has passed
the full same-scalar PES gates. The original radial-water canary's local
derivative does not override its rotation/torque failures. The distinct
fixed-box590 diagnostic now has one-equilibrium-geometry Cartesian and
orientation evidence plus a clean-commit seven-geometry/63-direction and
bidirectional cold/warm loop audit, plus a passing equilibrium
32/40/48/56-Angstrom box-tail audit. It also has a passing source-bound
20-molecule, 60-base-geometry and 11-path-point directional panel, followed by
a passing 20-reference-geometry, 465-component Cartesian panel. Those results
and residual-refinement evidence do not substitute for all-panel
symmetry/loop coverage, multi-geometry box convergence, Hessian, NVE,
physical-component validation, or public workflow admission.

## Solvation-energy interpretation

The current radial-GTO methane value is an **electrostatic C-PCM component**.
It excludes nonpolar/CDS and standard-state terms and therefore is not
`Delta G_solv`. Do not compare it directly with the FreeSolv experimental total
and do not append a CDS term from an incompatible cavity/profile.

When a Tier-E profile is eventually admitted, this guide will add complete
copy-paste MAPLE inputs and expected public output fields. OPT/NEB/TS/FREQ
examples will be added only after the corresponding F/H tiers pass; placeholder
commands that appear to work while bypassing admission are forbidden.

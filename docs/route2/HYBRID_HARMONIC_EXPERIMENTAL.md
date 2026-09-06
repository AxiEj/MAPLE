# Experimental MACE-MDP + MACE-POLAR harmonic Route 2

## What is admitted

Exactly one content-addressed profile is enabled:

```text
route2-profile-experimental-macemdppoint-macepolarinduced-
smoothharmonicgalerkin-electrostatic-v1
```

It admits only:

| tier | status | meaning |
| --- | --- | --- |
| E | yes | the registered operational electrostatic scalar |
| F | yes | the implicit-adjoint total derivative of that same scalar |
| H | no | no Hessian, HVP, FREQ, TS, or IRC admission |
| V | no | no common MACE-continuum variational functional claim |
| M | no | no MD or NVE admission |

The scalar is

\[
E(R)=E_{\rm vac}^{\rm MACE\text{-}POLAR}(R)
  -\frac12 b(R,c^*)^T A(R)^{-1}b(R,c^*),
\]

where the permanent source comes from unchanged MACE-MDP point monopoles and
dipoles, the induced source increment comes from MACE-POLAR, and `c*` is the
deterministic two-start operational root. This is a frozen energy ledger. It is
not presented as the stationary energy of the electronic update rule.

## Why the former GEPOL blocker is closed for this profile

The previous PCMSolver/GEPOL candidate could change its exposed active points
inside a Cartesian finite-difference stencil. The resulting force was not a
stable derivative of one smooth scalar.

This profile instead uses fixed-dimensional spherical-harmonic coefficient
spaces and smooth geometry-dependent coefficient assembly. No
laboratory-fixed surface grid or active-point deletion occurs in its
mathematical definition. The original numerical admission rebuilt every force
stencil point and:

1. rebuilds the complete harmonic operator;
2. solves both registered root starts;
3. checks root, charge, replay, and coefficient-topology invariants; and
4. evaluates the same registered scalar.

That structural change was followed by the missing same-profile numerical
tests. Two independent clean executions at candidate commit `4cf8db40`
returned identical measurement digest
`a28be11068866e035733c79064a7210739b31e17e3fbc979a5adaf22514a1a30`.

| gate | observed maximum |
| --- | ---: |
| old benzene failing-coordinate `h/4` difference | `3.941143707567041e-8 eV/A` |
| local Richardson force error | `1.4104595417549493e-5 eV/A` |
| independent water directional error | `2.513079225691106e-7 eV/A` |
| translation energy error | `0 eV` |
| translation force relative error | `0` |
| rotation energy error | `2.9654074751306325e-9 eV` |
| rotation force relative error | `2.4243672701543782e-8` |
| guarded closed-loop absolute work | `4.7405289175354166e-8 eV` |

The two raw records and their replicated admission record are under
`docs/route2/evidence/mace-mdp-polar-hybrid-harmonic-force-*4cf8db40.json`.

The daily force implementation now uses the matrix-free implicit adjoint of
the same scalar. The formula/sign audit was independently reviewed in a
verified ChatGPT Pro session, then checked against official checkpoints on CPU:

| analytic-force check | result |
| --- | ---: |
| water, all 9 components vs stored GPU Richardson | `9.5916e-9 eV/A` max abs |
| water directional Richardson-extrapolated mismatch | `6.417e-9 eV/A` |
| benzene component vs stored GPU Richardson | `6.036e-10 eV/A` |
| water translation force relative error | `2.243e-14` |
| water rotation force relative error | `3.285e-8` |
| water analytic midpoint closed-loop work | `2.136e-9 eV` |

The raw Pro transcript, mode proof, prompt, and real-checkpoint JSON records
are under
[`evidence/hybrid-harmonic-analytic-force-20260819/`](evidence/hybrid-harmonic-analytic-force-20260819/README.md).
Richardson remains callable as an independent diagnostic oracle; it is no
longer the normal MAPLE force path.

## Runtime contract

The low-level programmatic entry point remains:

```python
from maple.solvation.experimental import MACE_MDPPolarHybridSmoothHarmonicPES
```

For ordinary ASE energy/force calls, MAPLE also exposes the same admitted
profile as the built-in calculator model `macemdppolarhybrid`:

```python
from maple.function.calculator.set_calculator import SetCalculator

atoms.calc = SetCalculator(
    device="cuda",
    model="macemdppolarhybrid",
    output="maple.out",
    atoms=atoms,
).set_calculator()
energy_eV = atoms.get_potential_energy()
forces_eV_per_A = atoms.get_forces()
route2_identity = atoms.calc.results["route2"]
```

The calculator owns and verifies the two official checkpoint paths, constructs
the separately provenance-bound MACE-MDP permanent and MACE-POLAR responsive
adapters, applies the exact SMD-water Coulomb radii, and writes ASE-standard
`eV` and `eV/A` results. It exposes no Hessian/stress/virial property and does
not turn this electrostatic-only scalar into a complete solvation free energy.

At the MAPLE job boundary this opens ordinary single-point calculations and
first-order geometry optimization with `LBFGS`, `SD`, `SDCG`, or `CG`. The
dispatcher rejects `RFO`, `FREQ`, `TS`, `IRC`, `MD`, and scan jobs for this
calculator instead of leaking an E/F-only profile into a workflow that needs
an unadmitted Hessian or dynamical test.

Public evaluation is accepted only when the content-addressed hybrid
configuration and provenance
match the two frozen checkpoints, the analytic Gaussian-multipole evaluator,
the preregistered harmonic settings, SMD-water Coulomb radii, `float64`, and
`cuda`. Structurally compatible test doubles or altered parameters remain
usable through internal `solve/sample` methods but cannot publish an admitted
result. `evaluate(..., need_forces=True)` returns a `Route2Result` in ASE public
units (`eV`, `eV/A`).

The public force performs one central root solve plus a matrix-free transposed
adjoint solve and same-graph coordinate contractions. A call fails closed when
the true adjoint residual exceeds its frozen tolerance or any root, charge,
replay, or topology guard fails. The fourth-order Richardson implementation is
retained only for validation. There is no silent fallback to GEPOL.

## Scientific claim boundary

This admission means only that the implementation now exposes an experimental
conservative force for its declared scalar and narrow runtime domain. It does
not establish that the scalar is an accurate solution-phase free energy.

In particular, it currently represents conductor-limit electrostatics with
frozen water-cavity radii. It has no finite-dielectric solvent parameter, no
compatible nonpolar/CDS term, and no matched chemical-accuracy admission.
Consequently it must not be reported as complete `Delta G_solv`, generalized
to arbitrary solvents, or used for production FREQ/MD. First-order OPT is an
experimental same-scalar E/F workflow, not a chemical-accuracy admission.

The underlying electronic update also remains operational rather than Tier V.
Opening E/F does not change the existing counterevidence against a strict
common MACE-POLAR energy/source functional.

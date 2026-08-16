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
| F | yes | the numerical gradient of that same scalar |
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
mathematical definition. Every force stencil point:

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

## Runtime contract

The programmatic entry point is:

```python
from maple.solvation.experimental import MACE_MDPPolarHybridSmoothHarmonicPES
```

Callers must still construct the separately provenance-bound MACE-MDP
permanent and MACE-POLAR responsive adapters, then pass the exact element list,
cavity radii, dtype, and device to the PES constructor. Public evaluation is
accepted only when the content-addressed hybrid configuration and provenance
match the two frozen checkpoints, the analytic Gaussian-multipole evaluator,
the preregistered harmonic settings, SMD-water Coulomb radii, `float64`, and
`cuda`. Structurally compatible test doubles or altered parameters remain
usable through internal `solve/sample` methods but cannot publish an admitted
result. `evaluate(..., need_forces=True)` returns a `Route2Result` in ASE public
units (`eV`, `eV/A`).

The force is deliberately expensive: fourth-order Richardson differentiation
uses four fully re-solved scalar evaluations per Cartesian component. A call
fails closed if the local error estimate exceeds `2e-4 eV/A` or any root,
charge, replay, or topology guard fails. There is no silent fallback to GEPOL.

## Scientific claim boundary

This admission means only that the implementation now exposes an experimental
conservative force for its declared scalar and narrow runtime domain. It does
not establish that the scalar is an accurate solution-phase free energy.

In particular, it currently represents conductor-limit electrostatics with
frozen water-cavity radii. It has no finite-dielectric solvent parameter, no
compatible nonpolar/CDS term, and no matched chemical-accuracy admission.
Consequently it must not be reported as complete `Delta G_solv`, generalized
to arbitrary solvents, or used for production OPT/FREQ/MD.

The underlying electronic update also remains operational rather than Tier V.
Opening E/F does not change the existing counterevidence against a strict
common MACE-POLAR energy/source functional.

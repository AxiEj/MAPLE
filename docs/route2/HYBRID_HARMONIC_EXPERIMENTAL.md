# Experimental MACE-MDP + MACE-POLAR harmonic Route 2

## Current scientific mainline: general-source finite-dielectric v2

The older v1 surface documented below has a historical replicated E/F record at
commit `4cf8db40`. It is **not** admitted for the current modified provider
bytes and is not the current scientific mainline.
The mainline is the disabled, finite-dielectric general-source profile bound to

```text
route2-harmonic-ddpcm-general-source-psi-phi-primal-adjoint-v2
```

It preserves the heterogeneous source definition instead of applying one
Gaussian kernel to both branches:

* permanent MACE-MDP `q/p`: exterior point multipoles;
* induced MACE-POLAR increment: normalized 1.5-A Gaussian multipoles;
* MACE-POLAR drive: complete ddPCM phi-side adjoint in the native 1.5/3.0-A
  eight-channel field.

The general-source state stores both the ddPCM primal and adjoint coefficient
vectors.  Its source response is not assumed symmetric; the Phi0 energy uses
the exact symmetrized quadratic derivative, while the model-driving field has
its own exact JVP/VJP.

At the preregistered four-case fixed-geometry electrostatic diagnostic it
reports `0.8466964716 kcal/mol` MAE (`1.7191664851 kcal/mol` maximum), with all
cold/wide roots passing.  One real-checkpoint water force direction converges
as `1.69e-6 -> 4.22e-7 -> 1.06e-7 eV/A` under `h -> h/2 -> h/4`.

These results repair the former source-kernel mismatch and establish a strong
candidate, but do not admit complete solvation accuracy, a global single-root
domain, public analytic E/F, H/V/M, or Tier V.  Evidence is under
`docs/route2/evidence/mace-mdp-polar-harmonic-ddpcm-general-source-*4cf8db40.json`.

## Historical E/F record and current closure

The historical record names this content-addressed profile:

```text
route2-profile-experimental-macemdppoint-macepolarinduced-
smoothharmonicgalerkin-electrostatic-v1
```

The historical artifact and current registry differ deliberately:

| tier | historical `4cf8db40` artifact | current registry |
| --- | --- | --- |
| E | yes | no |
| F | yes | no |
| H | no | no |
| V | no | no |
| M | no | no |

The current provider implementation has changed since the frozen evidence run,
and its whole-file provenance is part of the runtime identity. The historical
hashes must not be replaced manually. Reopening E/F requires the complete
preregistered admission protocol to pass in two clean processes on the final
tree.

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

Callers may construct the research PES for internal `solve/sample` diagnostics.
Public `evaluate()` remains fail-closed because the current registry is
disabled. The historical runtime was bound to the exact checkpoint/adaptor
configuration, analytic Gaussian-multipole evaluator, preregistered harmonic
settings, SMD-water Coulomb radii, `float64`, and `cuda`; structurally compatible
test doubles or altered parameters never carried admission.

The force is deliberately expensive: fourth-order Richardson differentiation
uses four fully re-solved scalar evaluations per Cartesian component. A call
fails closed if the local error estimate exceeds `2e-4 eV/A` or any root,
charge, replay, or topology guard fails. There is no silent fallback to GEPOL.

## Scientific claim boundary

The historical record means only that the exact `4cf8db40` implementation
passed its narrow experimental conservative-force protocol. It is not a current
admission and does not establish an accurate solution-phase free energy.

In particular, it currently represents conductor-limit electrostatics with
frozen water-cavity radii. It has no finite-dielectric solvent parameter, no
compatible nonpolar/CDS term, and no matched chemical-accuracy admission.
Consequently it must not be reported as complete `Delta G_solv`, generalized
to arbitrary solvents, or used for production OPT/FREQ/MD.

The underlying electronic update also remains operational rather than Tier V.
The historical E/F result does not change the existing counterevidence against a strict
common MACE-POLAR energy/source functional.

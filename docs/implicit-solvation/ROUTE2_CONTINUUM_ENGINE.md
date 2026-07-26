# Route-2 provider-neutral continuum-engine boundary

## Stage decision

The rejected JGP94--pyddx derivative canary does not authorize another
continuum solve, a local switching patch, or a public `pyscf-swig` profile.
It does expose a separate maintainability problem: the present pyddx wrapper
owns both provider policy and the complete ML--SCF/adjoint orchestration.
Adding another research provider at that boundary would duplicate the
fixed-point solve, energy identity, force-state reproducibility checks,
adjoint, CDS merge, and component ledger.

This stage therefore makes one **zero-behaviour-change** architectural move:

> Extract the provider-independent ML--SCF, fixed-point adjoint, and energy/
> force ledger into one internal engine. Keep every public option, numerical
> constant, pyddx construction, SMD CDS definition, output filename, and
> scientific status unchanged.

The extraction is not a new solvation model and is not evidence for improved
hydration-free-energy accuracy, smoothness, or speed.

## Why a new public continuum provider is blocked

The available upstream-backed candidates do not yet satisfy the complete
Route-2 PES contract:

| Candidate | Upstream capability | Current Route-2 blocker |
|---|---|---|
| pyddx 0.8.0 ddPCM | Open ddPCM energy, reaction map, adjoint, and analytic force implementation | The locked local torsion evidence and the zero-solve JGP94 preflight show geometry-dependent active-set changes. The rejected canary must not be retuned or rerun. |
| PySCF 2.13.1 SWIG/ISWIG IEFPCM | Published SWIG/ISWIG surface and analytic PCM nuclear gradients | `pyscf.solvent.pcm.gen_surface` removes points for which `weight * switch <= 1e-16`. The finite atom-centred Lebedev orientation failed the existing second-molecule rigid-rotation qualification. A subsequently locked 12-solve fixed-density discriminator also rejected ISWIG: its methanol and acetone rotation spans were `1.686x` and `1.263x` the SWIG controls, and both molecules changed surviving parent counts across orientations. |
| ddX ddCOSMO/ddPCM/ddLPB | Common open interface with electrostatic energies and forces | It does not by itself remove the finite angular-grid/active-set issue already observed through pyddx. |
| FIXPVA/CPCM | Published continuous electrostatic surface and exact analytic gradients | No maintained open callable implementation suitable for the MAPLE runtime boundary has been identified. It remains a literature oracle, not a provider. |
| PCMSolver/GePol | Existing default fixed-conformer energy path | Its public interface does not provide the complete same-energy coordinate derivative required by Route 2. |

Increasing a Lebedev order, averaging several laboratory-frame orientations,
or projecting out net torque can reduce a symptom but does not differentiate a
single rotation-covariant scalar. A molecule-following grid can remove rigid
laboratory-grid anisotropy, but it still needs a smooth geometry-dependent
cavity and the complete frame derivative. The failed preflight shows that the
current pyddx active set does not supply that missing property.

## Scientific invariant

At a fixed geometry, Route 2 solves

\[
\mathbf c^\star
=\mathcal M_{\mathbf R}\!\left(
\mathcal P_{\mathbf R}\mathbf c^\star
\right),
\]

where \(\mathbf c\) is the neutral MACE-POLAR \(l\le1\) density,
\(\mathcal P_{\mathbf R}\) is one provider-owned reaction-field map, and
\(\mathcal M\) is the unmodified MACE-POLAR field response. The energy ledger
remains

\[
\Delta G_\mathrm{solv}
=
\left[
E_\mathrm{MACE}^{\mathrm{intrinsic}}(\mathbf f^\star)
-E_\mathrm{MACE}^{\mathrm{gas}}
\right]
+\frac12(\mathbf c^\star)^\mathsf T\mathbf Q\mathbf f^\star
+G_\mathrm{CDS},
\qquad
\mathbf f^\star=\mathcal P_{\mathbf R}\mathbf c^\star .
\]

The numerical mixing coefficient is a root-finding choice only. It is not
part of the physical residual differentiated by the fixed-point adjoint. The
engine must therefore continue to linearize the **unmixed** residual

\[
\mathbf R(\mathbf c;\mathbf R)
=\Pi_0\!\left[
\mathbf c-\mathcal M_{\mathbf R}(
\mathcal P_{\mathbf R}\mathbf c)
\right],
\]

where \(\mathbf Q\) converts the external Cartesian field order into the raw
MACE density-dual order used by the physical half-coupling. The engine must
solve the converged fixed-point implicit-function adjoint

\[
\left(\partial_{\mathbf c}\mathbf R\right)^\mathsf T\boldsymbol\lambda
=\Pi_0\nabla_{\mathbf c}E,
\qquad
\frac{dE}{d\mathbf R}
=E_{\mathbf R}
-\boldsymbol\lambda^\mathsf T
\partial_{\mathbf R}\mathbf R,
\]

in the neutral tangent space, then combine exactly one provider-owned full
reaction-field coordinate VJP with the explicit MACE density-position VJP and
the matching CDS analytic gradient. This is not a claim that the learned
density/continuum coupling is the stationary point of a joint variational
functional.

## Ownership boundary

### Internal engine owns

1. reuse or evaluation of the gas MACE-POLAR state;
2. the ML--SCF fixed-point loop and convergence history;
3. neutral-density and reaction-field shape/finite-value checks;
4. the half-coupling/provider-energy identity;
5. force-state density and intrinsic-energy reproducibility checks;
6. the unmixed residual, matrix-free adjoint, and total continuum derivative;
7. the component-resolved solute-polarization, PCM-polarization, electrostatic,
   CDS, standard-state, and total-solvation ledger.

### Provider wrapper owns

1. parser/profile/domain validation and public fail-closed policy;
2. atom typing, cavity radii, dielectric, discretization, and solver settings;
3. construction of one same-energy reaction-field object;
4. provider/profile provenance and public scientific-status labels;
5. provider-specific cache identity and audit filenames;
6. selection of the matching SMD CDS evaluator.

The engine accepts a reaction-field factory and a CDS evaluator. It does not
import pyddx, PySCF SWIG, PCMSolver, a parser, or a MAPLE model loader. A
reaction-field object remains responsible for its forward map, discrete
adjoint, scalar polarization energy, full coordinate VJP, and runtime
provenance. Components from different electrostatic providers may not be
mixed. The engine is an internal implementation detail and is not re-exported
as a public MAPLE API.

## Frozen non-goals

This stage must not:

- add or accept a new `provider`, profile, parser option, solvent, charge
  state, or model checkpoint;
- change `mixing=1.0`, any SCF/adjoint/identity tolerance, ddPCM setting,
  radius, CDS functional, or standard-state convention;
- change `route2-ddpcm-result.json`, `route2-ddpcm-state.npz`, their schemas,
  or the public `SolvationResult`;
- call a real continuum runtime or MACE checkpoint merely to validate the
  refactor;
- claim a solution-phase PES, QM accuracy, broader applicability, or a speed
  improvement.

The existing synthetic nonzero-response finite-difference test is the stage's
energy/force oracle. The complete targeted provider/response/derivative suite
must pass before and after the extraction. A later smooth-provider stage must
be separately pre-registered and must pass its own scalar identity,
translation, rotation, local Cartesian/torsional refinement, and end-to-end
ML--SCF force gates before any public dispatch is considered.

## References

- B. G. Johnson, P. M. W. Gill, and J. A. Pople, “A rotationally invariant
  procedure for density functional calculations,” *Chemical Physics Letters*
  **220**, 377--384 (1994), DOI:
  [10.1016/0009-2614(94)00199-5](https://doi.org/10.1016/0009-2614(94)00199-5).
- A. W. Lange and J. M. Herbert, “A smooth, nonsingular, and faithful
  discretization scheme for polarizable continuum models: The switching/
  Gaussian approach,” *J. Chem. Phys.* **133**, 244111 (2010), DOI:
  [10.1063/1.3511297](https://doi.org/10.1063/1.3511297).
- A. W. Lange and J. M. Herbert, “Polarizable continuum reaction-field
  solvation models affording smooth potential energy surfaces,”
  *J. Phys. Chem. Lett.* **1**, 556--561 (2010), DOI:
  [10.1021/jz900282c](https://doi.org/10.1021/jz900282c).
- F. Lipparini, G. Scalmani, L. Lagardère, B. Stamm, E. Cancès, Y. Maday,
  J.-P. Piquemal, M. J. Frisch, and B. Mennucci, “Quantum, classical, and
  hybrid QM/MM calculations in solution: General implementation of the
  ddCOSMO linear scaling strategy,” *J. Chem. Phys.* **141**, 184108 (2014),
  DOI: [10.1063/1.4901304](https://doi.org/10.1063/1.4901304).
- PySCF solvent documentation and source:
  [solvent models](https://pyscf.org/user/solvent.html) and
  [`pyscf.solvent.pcm`](https://pyscf.org/_modules/pyscf/solvent/pcm.html).
- ddX documentation:
  [ddX continuum-solvation library](https://ddsolvation.github.io/ddX/).

# Molecular IRC path and termination contract

This document defines the common path ledger used by MAPLE's molecular
`gs`, `lqa`, `hpc`, and `eulerpc` IRC integrators. It is deliberately narrower
than a claim that a chemically correct minimum-energy path has been found.

## Scope and unit boundary

IRC remains a legacy MAPLE job and therefore runs behind
`LegacyHartreeJobView`:

- energies are in Hartree;
- Cartesian coordinates are in Angstrom;
- Cartesian forces are in Hartree/Angstrom;
- Cartesian Hessians are in Hartree/Angstrom squared.

The input keys `f_max_th` and `f_rms_th` are consequently interpreted in
Hartree/Angstrom. Their retained MAPLE defaults are respectively
`2.0e-3` and `5.0e-4`; `max_steps=50` is the retained per-direction step limit.
The dispatcher does not replace explicit IRC thresholds with the unrelated
geometry-optimization `level=` table.

The TS/path energy comparison uses the explicit absolute
`path_energy_tolerance_hartree`, whose fail-closed default is `1.0e-7` Hartree.
MAPLE does not infer a backend's numerical precision from the magnitude of its
total energy. A lower-precision backend that cannot resolve this comparison
must supply and report a justified tolerance explicitly; doing so records that
the TS is a maximum only **within that tolerance**, not a strict maximum.

The numerical equality of the two force defaults to ORCA's IRC values must not
be confused with equality of physical units. ORCA 6.1.1 documents
`TolMaxG=2.0e-3` and `TolRMSG=5.0e-4` in atomic units, hence in Hartree/Bohr.
Using the 2022 CODATA Bohr radius,

\[
1\ E_h/a_0 = 1.88972612546\ E_h/\mathring{\mathrm A},
\]

so those ORCA defaults correspond to approximately `3.779452e-3` and
`9.448631e-4` Hartree/Angstrom. MAPLE does not silently reinterpret or loosen
its established Cartesian thresholds to make a particular example converge.

## Starting-point admission

Before either direction is propagated, the shared preflight requires:

1. a molecular, non-periodic, unconstrained geometry;
2. a force-stationary point under the configured tolerance;
3. a finite, symmetric Cartesian Hessian;
4. acceptable translation/rotation residuals after mass-metric projection;
5. exactly one robust negative vibrational mode;
6. positive remaining vibrational curvatures.

The exact validated transition-state energy, forces, and coordinates are saved
as a dedicated record. A displaced IRC image is never relabelled as the TS.

## Branch termination

Each requested direction returns an explicit status containing:

- `converged`;
- `termination_reason`;
- attempted and accepted macro-step counts;
- final maximum and RMS Cartesian forces;
- the two thresholds used by that run.

A branch is force-converged only when **both**

\[
\max_i |F_i| \le f_{\max}
\quad\text{and}\quad
\sqrt{(3N)^{-1}\sum_i F_i^2} \le f_{\mathrm{RMS}}
\]

hold at its final recorded point. Reaching `maximum_steps`, a vanishing
integration step, a stalled predictor, or another numerical stop is not
reported as convergence. By default, `require_converged_endpoints=true` raises
an admission error if either direction misses the two force criteria; setting
it to `false` retains the diagnostic path but does not upgrade its status.

## Path assembly and files

The common assembler validates that both branches refer to the same TS, that
no branch image duplicates its coordinates, and that no accepted path image
exceeds the TS energy beyond the explicitly reported numerical tolerance. The
combined order is deterministic for the canonical run-local eigenvector sign:

```text
forward endpoint ... forward first displacement -> exact TS
    -> backward first displacement ... backward endpoint
```

The exact TS occurs once in the combined record. The lower endpoint energy is
used only as the relative-energy reference; it never reverses the trajectory.
Separate forward/backward files begin at the exact TS and proceed outward.

Trajectories use extended XYZ. Because the private IRC energy is Hartree rather
than ASE's public eV, it is stored under the explicit `energy_hartree` metadata
key instead of the reserved ASE `energy` key.

## What a passing result does not prove

Force convergence of both legs and a valid path ledger still do **not** prove:

- that either endpoint is a positive-definite minimum;
- that the endpoints have the intended molecular connectivity;
- convergence with respect to IRC step length or integrator controls;
- agreement between the four numerical integration schemes;
- global smoothness across calculator neighbour-list or solvent-cavity events.

Chemical admission therefore still requires independent endpoint optimization
and frequency analysis at the same model level, structural/connectivity checks,
and a step-size study. Normal program termination alone is not such evidence.

## Separation from Route 2

The always-on propagation regression uses a model-independent analytic
translation/rotation-invariant double-well potential. It exercises all four
integrators without treating any ML checkpoint as scientific validation.

The retained ANI-1xnr example is only an opt-in **generic MAPLE IRC canary**.
It tests the shared task infrastructure and is not AIMNet2/pyddx evidence. It
cannot change any Route-2 energy, force, OPT, Hessian/FREQ/TS/IRC, or MD
admission flag. Route-2 admission remains governed by the scalar, reciprocity,
metric, coordinate-derivative, topology, rotation, and real-stack gates in
[`docs/route2`](route2/USER_GUIDE.md).

## References

- K. Fukui, *The Path of Chemical Reactions - The IRC Approach*, Accounts of
  Chemical Research **14** (1981) 363-368,
  <https://doi.org/10.1021/ar00072a001>.
- C. Gonzalez and H. B. Schlegel, *Reaction path following in mass-weighted
  internal coordinates*, Journal of Physical Chemistry **94** (1990)
  5523-5527, <https://doi.org/10.1021/j100377a021>.
- S. Ishida, K. Morokuma, and A. Komornicki, *The intrinsic reaction coordinate.
  An ab initio calculation for HNC -> HCN and H- + CH4 -> CH4 + H-*, Journal of
  Chemical Physics **66** (1977) 2153-2156,
  <https://doi.org/10.1063/1.434152>.
- ORCA 6.1.1 manual, *Intrinsic Reaction Coordinate*,
  <https://orca-manual.mpi-muelheim.mpg.de/contents/structurereactivity/irc.html>.
- P. J. Mohr et al., *CODATA recommended values of the fundamental physical
  constants: 2022*, Journal of Physical and Chemical Reference Data **54**
  (2025) 033105, <https://doi.org/10.1063/5.0279860>.

The references establish the reaction-path interpretation, representative
integration methods, and the unit conversion. They do not certify MAPLE's
implementations or replace the repository's numerical validation gates.

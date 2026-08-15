# MACE-MDP permanent source + MACE-POLAR induced response trial

## Status

This is a **research-only operational branch**. It does not admit Route-2
`E/F/H/V/M`, does not establish a unified variational functional, and does not
define a complete solvation free energy.

Frozen implementation/evidence commit:

```text
76422f4ee742d288e26072403030bf7477f8f145
```

Formal four-case artifact:

```text
/home/axie/MAPLE/MAPLE-implicitsolv-route2/.omx/benchmarks/
  route2-mace-mdp-polar-hybrid-pcmsolver-four-v1/formal-76422f4e.json
SHA256 144481feef854ee7b7cacaeda3951298d06f5dd4e3bdf4bada6da129bdb06541
```

The artifact records a clean execution tree, both checkpoint hashes, the
PCMSolver shared-library hash, all frozen input hashes, root histories, and two
independent initial fields per molecule.

## Implemented decomposition

The hybrid source is

\[
c_{\rm hybrid}(R,u)
=c_{\rm MDP}(R)
+\left[c_{\rm POLAR}(R,u)-c_{\rm POLAR}(R,0)\right].
\]

The two terms are not forced through one false source kernel:

* `c_MDP` is evaluated as exterior point monopoles/dipoles;
* the MACE-POLAR induced increment uses its normalized 1.5-A Gaussian source;
* the PCMSolver surface charge is returned through the audited MACE-POLAR
  1.5/3.0-A radial receiver.

Thus the operational equations used by the trial are

\[
v_\Gamma
=B_{\rm point}c_{\rm MDP}
+B_{\rm GTO}\Delta c_{\rm POLAR}(u),
\]

\[
q_\Gamma=Q_{\rm PCM}v_\Gamma,
\qquad
u=L_{\rm radial}q_\Gamma.
\]

The implementation is intentionally not accepted by the older single-`B`
`SeparatedElectronicResponseProvider` contract. The permanent point kernel
and induced Gaussian kernel must remain distinguishable.

Relevant files:

```text
maple/solvation/models/mace_mdp_polar_hybrid.py
tools/route2_release/run_mace_mdp_polar_hybrid_pcmsolver_panel.py
tests/route2_vnext/test_mace_mdp_polar_hybrid.py
```

## Frozen water-panel result

All values below are absolute errors in the matched PCMSolver polarization
component, in kcal/mol.

| Molecule | fixed MACE-MDP | hybrid self-consistent | change (positive is better) |
| --- | ---: | ---: | ---: |
| acetic acid | 0.764056 | 0.260126 | +0.503930 |
| benzene | 0.894633 | 1.296047 | -0.401413 |
| 2-acetoxyethyl acetate | 1.823863 | 1.600260 | +0.223602 |
| acetone | 0.285082 | 0.001117 | +0.283965 |

Aggregate:

```text
fixed MACE-MDP mean absolute error    0.941909 kcal/mol
hybrid mean absolute error            0.789388 kcal/mol
relative mean-error reduction          16.19 %

fixed MACE-MDP maximum error          1.823863 kcal/mol
hybrid maximum error                  1.600260 kcal/mol

hybrid improved cases                 3 / 4
hybrid worsened cases                 1 / 4
hybrid cases below 1 kcal/mol         2 / 4
```

The hybrid therefore shows a real average improvement on this very small
panel, but **not a uniform improvement**. In particular, benzene is a direct
counterexample to the claim that adding self-consistent response must always
beat the permanent-source post-processing model.

All four fixed points converged without damping in 8--11 iterations. Starting
from zero field and from twice the permanent-source reaction field produced a
maximum field difference of

```text
2.369e-12 eV
```

on this panel. This is useful local evidence, not a global single-root proof.

## Rejected alternative learned during the trial

A preliminary acetone probe also forced the MACE-MDP permanent moments through
the 1.5-A Gaussian source kernel. Its error was approximately
`5.82 kcal/mol`, compared with `0.00112 kcal/mol` for point-permanent plus
Gaussian-induced coupling. Therefore the all-Gaussian hybrid was rejected
before the formal panel. This preliminary comparison was not preregistered and
is not part of the formal accuracy artifact.

## Different-solvent scope

The electronic hybrid is not water-specific. The reaction operator can use a
different dielectric, so the same acetone state was probed on one fixed cavity
at four static dielectric constants:

| dielectric | iterations | polarization energy (kcal/mol) |
| ---: | ---: | ---: |
| 2.000 | 7 | -2.844653 |
| 4.806 | 8 | -4.938490 |
| 20.493 | 8 | -6.349926 |
| 78.355 | 9 | -6.642222 |

Every root reached a residual below `1e-10 eV`, and the response varied
monotonically with dielectric strength. The unregistered local probe JSON had
SHA256
`6cdb333db807b0bcea8e0903bc6e8fdf3b98c59f5476e6eb057e90c23c62fd11`.

This establishes only **electrostatic dielectric parameterization on a fixed
cavity**. A named-solvent calculation additionally requires a versioned cavity
and radii policy and, for total solvation free energy, a compatible nonpolar
term. Those parts were not tested here.

## Remaining blockers

1. **No complete energy ledger.** The panel compares the stationary continuum
   polarization component only. It does not decide whether a field-conditioned
   MACE-POLAR energy difference belongs in the operational scalar.
2. **No total coordinate derivative.** MACE-MDP atom-resolved `q/p` coordinate
   derivatives and the combined point/GTO/cavity coordinate VJP are not yet
   available. The hybrid force API therefore fails closed.
3. **No strict Tier V.** The MACE-MDP permanent source and original MACE-POLAR
   induced source are not derived from one scalar graph.
4. **No structural rotation proof for this backend.** The trial intentionally
   reused the frozen PCMSolver/GEPOL cases for matched comparison; it did not
   replace them with the harmonic-Galerkin continuum.
5. **Insufficient accuracy coverage.** Four molecules, with one worsening case,
   cannot justify a quantitative production claim.
6. **No named-solvent transfer panel.** Only water references exist; the
   dielectric sweep has no matched QM reference and keeps the water cavity.

## Next bounded decision

The next useful experiment is not another solver rewrite. It is a larger,
frozen matched-component panel using this exact point-permanent/GTO-induced
identity. If the improvement remains nonuniform or small, this hybrid should
remain a research branch rather than being promoted by tuning response scales
against solvation energies. If it survives that gate, the next engineering
task is the distinct-kernel coordinate VJP and an operational ledger audit.


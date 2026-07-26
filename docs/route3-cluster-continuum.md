# Route 3: supermolecule + implicit outer solvent

Following Section 4.7 of Cramer and Truhlar, Route 3 is MAPLE's experimental
**supermolecule approach**:

```text
solute + first-shell explicit solvent + outer continuum
```

The solute and explicit first-shell molecules are treated as one
supermolecule, which is then surrounded by continuum solvent. In modern
terminology this is also a member of the broader cluster–continuum or
combined discrete/SCRF family.

The same complete non-periodic supermolecule is passed to both terms:

```text
E_total = E_inner_MLIP(cluster) + DeltaE_outer(cluster)
```

For the optional TBLite route,

```text
DeltaE_outer =
    E_GFN2-xTB/ALPB(cluster) - E_GFN2-xTB/vacuum(cluster)
```

and forces use the corresponding difference. The vacuum subtraction keeps the
GFN2-xTB intracluster energy out of the final result; the selected MAPLE MLIP
remains the inner potential.

## Why this is a generic supermolecule composition layer

`ClusterContinuumCalculator` is retained as the implementation class name. It
wraps the calculator returned by
`SetCalculator`. It does not add solvent code to ANI, AIMNet2, MACE, UMA, or a
custom `module=` calculator. Therefore a new inner MLIP needs only to satisfy
MAPLE's existing calculator protocol and Hartree result-unit contract.

The wrapper records:

- `inner_energy`: MLIP energy of the full cluster;
- `outer_correction`: outer continuum delta;
- `energy` / `free_energy`: their sum;
- component and total forces when both contributors support forces.

Single-point output prints the two energy components before the total.

## Supported outer providers

| Input | Outer contribution | Public capability |
|---|---|---|
| `method=gbsa` | Existing MAPLE heuristic GB-polar/QEq correction | Energy only; `#sp(verbose=0)` |
| `method=alpb,provider=tblite` | GFN2-xTB ALPB minus GFN2-xTB vacuum | Energy and forces; numerical total Hessian/HVP |

Both paths require `experimental=true` and reject periodic input.

The first GB route is an engineering probe, not production GBSA/OBC-II: its
nonpolar surface-area term is absent and its geometry-dependent QEq charges are
not variationally coupled to the solvent energy.

The TBLite provider is optional and is not a MAPLE core dependency. It uses
`GFN2-xTB`, named-solvent ALPB, and the `gsolv` reference state in this first
implementation. Install an upstream TBLite build exposing
`tblite.ase.TBLite` (for example, `conda install -c conda-forge tblite`);
MAPLE otherwise fails before the inner model is loaded.

## Input modes

### Build a first-shell coordinate guess

```text
#model=macepolm
#sp(verbose=0)
#device=cpu
#solv(explicit=water,number=4,implicit=water,method=gbsa,experimental=true)
```

`explicit=water,number=4` runs MAPLE's existing cluster builder before the
calculator is initialized. The builder is a coordinate initializer, not a
reaction-center optimizer or hydrogen-bond-network sampler.

### Use a prebuilt microsolvated cluster

Supply the solute and selected first-shell molecules directly, and omit
`explicit=`:

```text
#model=macepolm
#opt(method=lbfgs)
#device=cpu
#solv(implicit=water,method=alpb,provider=tblite,experimental=true)
```

This is the preferred starting point for proton transfer, hydrolysis,
SN1/SN2, metal coordination, and other cases where solvent placement changes
the reaction coordinate.

Runnable input templates are under `examples/solvation/route3/`.

## Inner MLIP selection

The composition layer is model-independent. Selection remains a scientific
domain decision:

- MACE-POLAR is the current default candidate where charge/spin-aware molecular
  coverage is appropriate.
- UMA `omol` is a useful independent cross-check, especially before trusting
  metal-containing clusters.
- MACE-O-MOL is a useful independent organic reaction/transition-state
  comparison.
- AIMNet2/AIMNet2-NSE can be auxiliary checks within their supported domain.
- ANI and MACE-OFF are baselines only where their element and chemistry
  coverage is adequate.

Custom models should use MAPLE's existing `module=` calculator plug-in path;
Route 3 does not require a model-specific adapter.

## Scientific and thermodynamic boundaries

1. Keep atom ordering and explicit solvent count fixed across reactant,
   transition state, and product. Changing cluster stoichiometry requires a
   separate thermodynamic cycle and is not implemented here.
2. MAPLE does not impose a hard 2–10-solvent limit. That range is a practical
   first-shell starting point, not an invariant.
3. A single cluster geometry is not a converged solution free energy.
   Microsolvation-state and conformer sampling remain the user's responsibility.
4. `free_energy` is the ASE alias of the composed potential energy. It is not
   BAR/FEP hydration free energy.
5. Do not combine a solvent-conditioned MLIP (for example, a model whose
   prediction already includes continuum conditioning) with this outer
   correction unless its published energy decomposition proves that the terms
   are non-overlapping.
6. The TBLite delta is an approximate response correction. It is not a
   self-consistent PCM built from the MLIP electron density.

## References

- C. J. Cramer and D. G. Truhlar, “Implicit Solvation Models: Equilibria,
  Structure, Spectra, and Dynamics,” *Chem. Rev.* **1999**, 99, 2161–2200,
  especially Section 4.7, “Explicit Solvent in the First Solvation Shell.”
  <https://doi.org/10.1021/cr960149m>
- S. Ehlert, M. Stahn, S. Spicher, S. Grimme, “Robust and Efficient Implicit
  Solvation Model for Fast Semiempirical Methods,” *J. Chem. Theory Comput.*
  **2021**, 17, 4250–4261. <https://doi.org/10.1021/acs.jctc.1c00471>
- TBLite ASE interface:
  <https://tblite.readthedocs.io/en/latest/users/ase.html>
- TBLite solvation specification:
  <https://tblite.readthedocs.io/en/latest/spec/solvation.html>

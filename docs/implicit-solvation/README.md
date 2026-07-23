# MAPLE implicit solvation: Route 2

This branch contains only the MACE-POLAR + SMD/IEFPCM route. It does not
contain the fixed-charge PB/GB implementation. All calculations require
`experimental=true` until the frozen scientific gates pass.

## Route-2 contract: MACE-POLAR + SMD

Route 2 is separate from the fixed-charge PB/GB path. It couples the official
pretrained MACE-POLAR-1-M coarse-grained charge moments to an external
PCMSolver IEFPCM reaction field and MAPLE's native aqueous SMD CDS term. At the
dielectric boundary, the moments use their cavity-exterior point-multipole
expansion rather than extending MACE's internal 1.5 A GTO smearing across the
cavity. It
does not train or fine-tune a model, consume MOL2 partial charges, invoke a
quantum-chemistry executable, or use a solvation-trained MLIP.

The public v1 input is:

```text
#model=macepol-m
#sp
#solv(implicit=water,method=smd,response=scf,standard_state=1m,experimental=true)

0 1
MOL2 molecule.mol2
```

The omitted locked defaults are `provider=pcmsolver` and
`profile=smd-iefpcm`.  `response=scf` is the public default;
`response=frozen` is retained only as a diagnostic that solves PCM once from
the gas-phase density.  There is no public mock or ddX backend.

### Route-2 runtime

Install the exact MACE release used by the contract and let its official
foundation-model loader populate the upstream cache:

```bash
pip install 'mace-torch==0.3.16'
```

Install PCMSolver separately.  MAPLE neither vendors nor builds PCMSolver and
requires both the matching official Python input parser and the v1.1.12-style
`libpcm.so` C API:

```bash
export PCMSOLVER_LIBRARY=/absolute/path/to/libpcm.so
export PCMSOLVER_PYTHON_PATH=/absolute/path/to/pcmsolver/lib/python
```

When explicit paths are used, MAPLE treats `PCMSOLVER_LIBRARY` as
authoritative and never falls back to another soname. The parser must resolve
under the same installation prefix as that library; mixed installations fail
before cavity construction.

The model is loaded as `polar-1-m` in float64. MAPLE uses the pretrained
model's existing GTO field-response path, supplies a different reaction
potential/gradient at each atom, and changes no learned weight.  The official
weights remain subject to the upstream Academic Software License; MAPLE does
not bundle or mirror them.

### Route-2 result and domain

MAPLE reports three distinct quantities:

```text
Gas-phase MLIP energy
Solvation free-energy correction (Delta G_solv, 1M(gas)->1M(solution))
Combined E_MLIP(gas)+Delta G_solv
```

ASE's `free_energy` field is the same combined electronic-plus-solvation value,
not a thermochemical Gibbs free energy with vibrational or thermal terms. The
route uses a 1 M gas to 1 M solution convention, so it does **not** add the
commonly used 1 atm to 1 M `1.89 kcal/mol` correction.

The v1 domain is deliberately fail-closed:

- one fixed Tripos MOL2 conformer; single-point energy only;
- neutral closed-shell molecules (`0 1`), no salts, zwitterions, radicals, or
  periodic cells;
- H/C/N/O/F/P/S/Cl/Br/I and molecular mass from 16 through 500 Da;
- water only; no forces, optimization, Hessian, scan, or MD;
- exact official MACE-POLAR-1-M only; `#charge`, D4, other models, and
  alternate continuum providers are rejected.

Every evaluation retains `manifest.json`, the human and parsed PCMSolver
inputs, PCMSolver/PEDRA cavity side files, `route2-state.npz`, and
`route2-result.json` under `<output>.implicit/`; legacy provider files are
contained there rather than written into the launch directory. The provenance
labels the model output correctly as a coarse-grained net charge density
rather than a QM electron density and keeps `accuracy_certified=false` until
the frozen benchmark passes.

See [FORMULAS_AND_REFERENCES.md](FORMULAS_AND_REFERENCES.md) for equations and
the literature ledger, and [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for the
passing engineering checks and still-open scientific gates.

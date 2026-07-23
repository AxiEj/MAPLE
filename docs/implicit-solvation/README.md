# MAPLE implicit solvation

MAPLE composes an MLIP gas-phase potential with an additive, auditable solvent
correction.  The first implemented validation domain is one neutral, closed-shell,
connected organic molecule in water, supplied as a Tripos MOL2 file with
explicit bonds and hydrogens.

All implicit-solvation providers currently require `experimental=true`.  This
flag records that the implementation gates pass locally while the public
scientific benchmark gate remains open; it is not a request to mix arbitrary
parameter profiles.

## Input contract

MOL2 fixed charges (RESP, RESP2, or another documented charge set):

```text
#model=aimnet2
#sp(verbose=1)
#charge(source=mol2,label=resp2)
#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,experimental=true)

0 1
MOL2 molecule.mol2
```

MAPLE-orchestrated charge generation:

```text
#charge(source=maple,method=am1bcc,geometry=keep)
#charge(source=maple,method=abcg2,geometry=keep)
#charge(source=maple,method=qeq-gto,mode=fixed)
#charge(source=maple,method=qeq-gto,mode=polarizable)
```

QEq/CQEq are frozen experimental research profiles.  They require explicit
`method=qeq-gto` selection, are never selected as defaults or provider
fallbacks, and are not accuracy-certified.  Current route-1 certification work
is limited to MOL2 fixed charges, AM1-BCC, and ABCG2.

`qeq-gto` performs the full published hydrogen SCF update for both the
idempotential and screening exponent in fixed mode.  `mode=polarizable` does
not reuse that nonvariational fixed-point equation: it switches to the
consistent-QEq (CQEq) derivative, solves the nonlinear charge-constrained
minimum of CQEq plus the GB polar energy, and applies the envelope theorem only
after KKT and projected-Hessian minimum gates pass.

AmberTools is an optional executable provider.  It can be kept outside the
main MAPLE environment to avoid dependency conflicts:

```bash
conda create -n maple-ambertools --override-channels -c conda-forge python=3.11 'ambertools=26.0'
```

Either activate that environment before running MAPLE or point the charge
provider at its Antechamber wrapper explicitly:

```text
#charge(source=maple,method=am1bcc,geometry=keep,executable=/path/to/maple-ambertools/bin/antechamber)
#charge(source=maple,method=abcg2,geometry=keep,executable=/path/to/maple-ambertools/bin/antechamber)
```

`geometry=keep` always preserves the submitted MOL2 geometry.  Antechamber
outputs, commands, stdout, stderr, atom mapping, and charge sums are retained in
`<output>.implicit/`.  `geometry=provider` explicitly adopts the geometry
written by the provider.  Fixed charges are generated/read once and remain
frozen through SP, OPT, and SCAN/PES.

MOL2 charges are never silently normalized.  Their sum must match the declared
molecular charge within `1e-4 e`.

## GB methods

| MAPLE model | Amber selector | locked radii profile | provider |
|---|---:|---|---|
| `hct` | `igb=1` | `hct-mbondi` | OpenMM `GBSAHCTForce` |
| `obc1` | `igb=2` | `obc1-mbondi2` | OpenMM `GBSAOBC1Force` |
| `obc2` | `igb=5` | `obc2-mbondi2` | OpenMM `GBSAOBC2Force` |
| `gbn` | `igb=7` | `gbn-bondi` | OpenMM `GBSAGBnForce` |
| `gbn2` | `igb=8` | `gbn2-mbondi3` | OpenMM `GBSAGBn2Force` |

OBC-II is the temporary default.  `nonpolar=ace` is the default;
`nonpolar=lcpo` requires OpenMM 8.5 or newer.  `nonpolar=none` is diagnostic
only and requires `experimental=true`.  OpenMM returns both correction energy
and conservative correction force, so GB is available to SP, OPT, and SCAN/PES.

`nonpolar=ace` and Amber `gbsa=1` are not aliases: Amber's latter selector is
LCPO.  MAPLE uses LCPO only for independent Amber complete-energy/force parity
and keeps ACE as the predeclared experimental-accuracy profile.  OpenMM 8.5.2
does not expose Amber's phosphorus-specific GBn2 alpha/beta/gamma parameters;
MAPLE therefore rejects `model=gbn2` for P-containing molecules instead of
silently using the generic fallback.  The other four GB models remain available.

## PB methods

```text
#charge(source=mol2,label=resp2)
#solv(implicit=water,method=pb,model=lpb,provider=apbs,profile=generic-mbondi2,nonpolar=apbs,experimental=true)
```

The APBS provider writes PQR, evaluates documented solvated/reference LPBE
blocks, subtracts their electrostatic energies, and adds an APBS APOLAR term.
The locked generic nonpolar profile is the APBS-style SASA reduction
`gamma*A` with `gamma=0.105 kJ mol^-1 A^-2`, zero pressure, and zero bulk
solvent density.  The default `97^3` grid at `0.33 A` follows the APBS solvation
example; alternative `grid_points` must have the nlev=4 form `c*32+1`, and
MAPLE refuses grids that do not enclose the molecular surface plus the SPL2
boundary margin.
PB is SP-energy-only until its grid-converged force path passes the independent
force gate.

The `amber-pbsa/abcg2-pbsa-2023` configuration is parsed and pairing-locked but
execution deliberately stops at an evidence gate.  The paper reports optimized
GAFF2 atom-type radii (including new `on`, `oi`, `hn1`, `hn2`, `hn3` types) and
a refitted nonpolar model; those exact redistributable parameter artifacts have
not been obtained from an authoritative upstream package.  MAPLE will not
invent or approximate them under the certified profile name.

## Result composition

For fixed charges:

```text
E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)
```

For `qeq-gto,mode=fixed`, QEq is evaluated once at the submitted reference
geometry and those charges remain fixed during SP, OPT, and SCAN/PES, matching
the lifecycle used for AM1-BCC, ABCG2, and MOL2-provided charges.

For `qeq-gto,mode=polarizable`, MAPLE uses the literature-defined consistent
QEq derivative at every geometry:

```text
q_vac  = argmin_q E_CQEq(R,q),                         sum(q)=Q
q_solv = argmin_q [E_CQEq(R,q) + G_GB,polar(R,q)],     sum(q)=Q
DeltaE = E_CQEq(R,q_solv) - E_CQEq(R,q_vac)
       + G_GB,polar(R,q_solv) + G_nonpolar(R)
```

The polarizable profile remains experimental: the mathematics and forces are
consistent, but the QEq and Amber GB parameters were not jointly fit.

## Route 2: MACE-POLAR + SMD

Route 2 is separate from the fixed-charge PB/GB path.  It couples the official
pretrained MACE-POLAR-1-M coarse-grained charge density to an external
PCMSolver IEFPCM reaction field and MAPLE's native aqueous SMD CDS term.  It
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

The model is loaded as `polar-1-m` in float64.  MAPLE uses the pretrained
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

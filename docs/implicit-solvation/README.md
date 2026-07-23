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

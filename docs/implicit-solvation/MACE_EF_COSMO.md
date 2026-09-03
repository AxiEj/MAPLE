# MACE-EF COSMO and Torch COSMO-RS route

## Scope

This branch keeps two different physical roles separate:

1. `torch-smooth-cosmo` is a differentiable conductor-limit continuum scalar
   coupled to MACE-EF. It provides the same diagnostic SP, analytic-force,
   L-BFGS OPT, finite-difference FREQ, and P-RFO TS surface as the smooth-ddPCM
   route.
2. `maple.function.cosmors_torch` is a fixed-structure statistical-
   thermodynamics layer. It contains a batched PyTorch COSMOspace solver and a
   provider-neutral kinetic-solvent-effect calculator. It is not added to the
   force loop or MD.

Neither layer copies COSMOtherm code or proprietary parameter files.

## MACE-EF conductor COSMO input

```text
#model=mace-polar-ef-v2(model_path=/absolute/path/macepol-ef-v2.pt)
#sp(verbose=1)
#device=gpu0
#solv(implicit=acetonitrile,method=cosmo,provider=torch-smooth-cosmo,profile=mace-polar-ef-v2-smooth-cosmo-l3-p6-r96-128-128-multisolv-derivatives-known-nonpassive-v1,response=scf,standard_state=1m,experimental=true,acknowledge_known_nonpassive=true,acknowledge_unvalidated_derivatives=true)

0 1
O   0.000000   0.000000   0.000000
H   0.957200   0.000000   0.000000
H  -0.239987   0.927297   0.000000
```

The conductor operator is represented by the largest finite binary64 value.
This makes `(epsilon+1)/(epsilon-1)` exactly one without introducing
`inf/inf`; tests require the assembled dielectric and conductor matrices to
be bitwise identical.

## COSMOspace Torch core

The first clean-room Torch block implements:

- published openCOSMO-RS 24a neutral interaction parameters;
- misfit and hydrogen-bond segment interaction matrices;
- batched successive-substitution COSMOspace roots;
- residual molecular activity from segment counts;
- Torch autograd through the solved iteration sequence.

Its frozen numerical oracle was produced with upstream
`TUHH-TVT/openCOSMO-RS_py` commit
`3db6614925ded5fc47a77d1e34f4a1975e518a9b` and is stored in
`benchmarks/opencosmors-python-cosmospace-oracle-v1.json`. Matching this core
does not yet establish full openCOSMO-RS 24a solvation-energy parity: ORCA
surface parsing, sigma averaging/clustering, combinatorial terms, molar-volume
handling, and the remaining solvation-energy terms must also be connected.

## Relative kinetic solvent effects

The KSE calculator supports any provenance-identified provider, including
COSMOtherm, openCOSMO-RS, or the future complete Torch backend. For every
solvent it evaluates

```text
delta_G_activation_solv = G_solv(TS) - sum_i nu_i G_solv(reactant_i)
```

and then

```text
ln(k_target/k_reference) =
    -(delta_G_activation_solv,target
      - delta_G_activation_solv,reference)/(R T)
```

This explicitly handles bimolecular reactions such as `CN- + CH3Br`; a
single combined reactant term is used only when the input deliberately
defines a pre-reaction complex.

Run:

```bash
maple-cosmors-kse kse-input.json kse-output.json
```

Input schema:

```json
{
  "temperature_k": 298.15,
  "standard_state": "1M",
  "provider_identity": "exact engine and parameterization identity",
  "target": {
    "solvent": "acetonitrile",
    "transition_state": {
      "species": "CN--CH3Br-TS",
      "delta_g_solvation_kcal_mol": -10.0
    },
    "reactants": [
      {"species": "CN-", "delta_g_solvation_kcal_mol": -5.0},
      {"species": "CH3Br", "delta_g_solvation_kcal_mol": -2.0}
    ]
  },
  "reference": {
    "solvent": "water",
    "transition_state": {
      "species": "CN--CH3Br-TS",
      "delta_g_solvation_kcal_mol": -8.0
    },
    "reactants": [
      {"species": "CN-", "delta_g_solvation_kcal_mol": -3.0},
      {"species": "CH3Br", "delta_g_solvation_kcal_mol": -1.0}
    ]
  }
}
```

The numeric values above demonstrate the schema only and are not chemical
predictions.

## Current ion boundary

The openCOSMO-RS 24a parameterization is a neutral-molecule model. A Torch
translation does not turn it into a validated anion model. Ionic SN2 work
must retain a separate COSMOtherm/ionic-reference identity until an ionic
parameterization and independent validation are available.

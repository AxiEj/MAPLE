# Route 2 MACE–continuum energy-functional audit

**Date:** 2026-07-26  
**Audited Route-2 head:** `d69049f8b5048c5c762d738f4151d54df822d310`  
**Status:** production-blocking theory/validation contract; no literal double-counting defect is asserted without the missing real-model tests.

## 1. Question

Route 2 currently reports

```text
Delta G_elec
  = E_MACE,local-field(V_reac) - E_MACE,gas
  + 1/2 <c, P c>,
```

where `c` is the converged MACE-POLAR `l<=1` density block and `P c` is the continuum reaction field.

The scientific question is whether the first difference is a field-independent solute internal/polarisation energy, so that adding the continuum half-coupling reproduces a standard self-consistent PCM energy without omission or double counting.

## 2. What the official MACE-POLAR model actually computes

The released `polar-1-m`/`polar-1-l` configuration has:

```text
add_local_electron_energy: True
field_readout_config.type: OneBodyMLPFieldReadout
```

The MACE-POLAR paper defines

```text
E_total = E_local + E_non-local + E_electrostatic,
```

with:

```text
E_electrostatic
  = 1/2 int int rho(r) rho(r') / |r-r'| dr dr'
  + int rho(r) v_app(r) dr.
```

It also states explicitly that `E_non-local` is a **field- and charge-dependent learned correction**.

The source implementation follows that definition:

1. the applied potential is projected into both spin-channel electrostatic features with a factor `1/2` per channel;
2. those potential features alter the density update;
3. the same potential features enter `OneBodyMLPFieldReadout`, producing `electron_energy`/`E_non-local`;
4. the final Gaussian Hartree energy is added;
5. the explicit applied-field coupling is added separately from the graph-level `external_potential` and total dipole.

## 3. What the MAPLE local-field adapter changes

MAPLE replaces the upstream external-field projector input with atom-indexed local values, but keeps the graph-level `external_field` equal to zero.

Consequently:

- the local reaction potential affects the density iterations;
- it affects the learned `electron_energy`/`E_non-local`;
- it affects the Gaussian internal Hartree energy through the changed density;
- the upstream explicit graph-level `E_app` term remains zero.

This supports one limited statement:

> The upstream explicit homogeneous-field coupling is not literally added once by MACE and then added a second time by MAPLE.

It does **not** support the stronger current statement:

> The returned local-field MACE energy is a field-independent intrinsic solute energy that rigorously excludes all explicit reaction-potential dependence.

The learned non-local readout has explicit potential-feature dependence by construction.

## 4. Why the standard PCM variational argument does not automatically apply

For a conventional variational solute model, one can define an internal functional `E0[c]` and continuum energy

```text
E_PCM[c] = E0[c] + 1/2 <c, P c>.
```

Stationarity gives the reaction potential in the electronic equations and eliminates the implicit density derivative from the final force expression.

MACE-POLAR-1 does not expose this structure:

- its density is predicted by a finite sequence of learned field updates, not by minimising the reported energy;
- the density was not supervised by QM density/ESP labels;
- `E_non-local` depends directly on field features;
- MAPLE's own real-model derivative work found that `dE_MACE/df` cannot be replaced by the returned density;
- Route 2 therefore uses an implicit-function adjoint for the algorithmic fixed point rather than a Hellmann–Feynman/stationary-energy simplification.

Thus the present implementation defines two objects:

1. a self-consistent response equation

```text
c = M(P c),
```

2. an operational scalar energy

```text
E_op(c) = E_MACE,no-explicit-Eapp(P c) - E_MACE,gas
          + 1/2 <c, P c>.
```

The code can differentiate `E_op` consistently through the fixed point. That is an important engineering achievement. It does not prove that `E_op` is the unique or physically correct PCM free-energy functional for this pretrained model.

## 5. Current verdict on double counting

### Not established

There is no source-level evidence that the exact same explicit `int rho V_reac` term is included fully inside the MAPLE local-field MACE energy and then added again as PCM work.

### Also not established

There is no proof that the MACE local-field energy difference is a pure solute internal polarisation cost compatible with the standard `1/2` PCM work formula.

### Required wording until validation

Use:

> `delta_e_model_response`: change in the selected MACE-POLAR scalar output when the Route-2 local reaction-potential features are applied.

Do not use without qualification:

> exact solute polarisation energy, intrinsic electronic energy, variational PCM solute energy, or SCF–PCM total energy.

The combined scalar should remain a **versioned operational Route-2 energy candidate**.

## 6. Required source changes for auditability

`PolarState` currently retains only total energy, density, dipole and optional forces. It discards the upstream component outputs.

Extend the state/audit schema to retain at least:

```text
interaction_energy
node/local energy contribution
electron_energy (learned E_non-local)
electrostatic_energy (internal Gaussian Hartree term)
explicit applied-potential energy
full total energy
```

The local-field adapter must report explicitly that the graph-level applied-potential term is zero.

Every SCF iteration and final state should record component deltas, not only the total scalar.

## 7. Mandatory real-model discriminators

### 7.1 Uniform-field semantic equivalence

A uniform field is the one non-zero external potential supported by the official public model contract.

For the same geometry and field:

1. evaluate the ordinary upstream graph-level external-field path;
2. evaluate the MAPLE atomwise local-projector path using the exactly corresponding barycentre-centred node potential and gradient;
3. compare density coefficients, dipole, `electron_energy`, internal Hartree energy, explicit applied-field term and total energy;
4. verify that the difference between full upstream total energy and the local-path scalar equals the exact upstream applied-potential component, within a preregistered tolerance;
5. repeat across field signs, directions and magnitudes.

This is the minimum proof that the local adapter has the intended energy semantics.

### 7.2 Component-wise constant-potential gauge test

For neutral molecules, shift all scalar node potentials by a constant while keeping gradients fixed. Track every component and final self-consistent correction. A failure cannot be hidden by cancellation in the total.

### 7.3 Charging-path test

For `lambda in [0,1]`, solve

```text
c_lambda = M(lambda P c_lambda)
```

and record:

```text
E_MACE,no-Eapp(lambda)
<c_lambda, P c_lambda>
dE_MACE,no-Eapp/dlambda
```

Compare the final operational energy with numerical thermodynamic integration along the same algorithmic path. This does not by itself prove QM correctness, but it determines whether the chosen half-coupling has an internally consistent charging interpretation.

### 7.4 QM PCM component benchmark

On a fixed same-geometry panel, compare against a maintained QM PCM implementation:

- gas and solution electronic energies;
- solute distortion/polarisation term;
- continuum polarisation work;
- total electrostatic correction;
- cavity MEP and reaction field;
- field-induced dipole.

Because component errors already cancel in the existing bounded panel, total `Delta G` alone is not an acceptable discriminator.

### 7.5 Alternative energy ledgers must be preregistered

Before seeing confirmation results, compare at least:

```text
A: delta E_MACE,no-Eapp + 1/2 <c,Pc>   # current
B: delta E_MACE,full-applied-field - 1/2 <c,Pc>
C: charging-path thermodynamic integral
```

Only ledgers that can be computed with an exact, documented source/field pairing should enter. Do not select a ledger by minimising final FreeSolv error after inspection.

## 8. Force implication

The current adjoint differentiates the operational scalar that the code implements, provided the component graph is unchanged and all geometry transforms are included. Therefore a finite-difference force match establishes **energy–force consistency for that operational scalar**.

It does not establish:

- variational equivalence to conventional PCM;
- absence of model-specific field-energy contamination;
- QM component fidelity;
- thermodynamic free-energy correctness.

These claims require the discriminators above.

## 9. Release gate

Route 2 must remain experimental and must not be described as a production self-consistent MLIP–PCM energy functional until:

- the component-resolved uniform-field semantic test passes;
- the checkpoint-specific energy components are archived;
- a charging-path identity is established;
- the current ledger wins a preregistered QM component comparison;
- the P0 SMD chemistry defects and all derivative/provenance blockers in the parent audits are corrected.

## 10. Primary references inspected

- MACE-POLAR-1 paper, Eqs. 1, 16–17, 28–34 and external-field response section.
- Official `mace_polar_1/config-mace-polar-1.yaml` verified against released checkpoints.
- `mace/modules/extensions.py`, `PolarMACE.forward()`.
- `mace/modules/field_blocks.py`, `OneBodyMLPFieldReadout`.
- MAPLE `_LocalReactionFieldProjector`, `polar_state()`, Route-2 engine and derivative code.

# Route 2 hybrid literature refresh — 2026-08-27

This note records a target-independent literature check made while the frozen
VQM24 observable batch was running.  It does not change any preregistered data
split, target, or gate.

## 1. Baldwin et al., *Design Space of Self-Consistent Electrostatic MLIPs*

- Preprint: https://arxiv.org/abs/2603.14700
- Current version inspected: v2, 8 August 2026.

The paper's coarse-grained DFT analysis requires fidelity of the reference
coarse-grained density, fidelity of its first-order response, and control of
higher-order response.  Its MACE implementations use atom-centred Gaussian
multipoles, normally `sigma=1.5 A` and `l<=1`; it reports little gain from
higher multipoles relative to their cost.  It also distinguishes the source
basis from the richer potential-feature basis and makes the applied-field work
an explicit term of the population-constrained scalar.

This directly supports the current MAPLE sequence:

1. dense zero-field full-MEP fit/audit tests the reference source span;
2. nonuniform finite-field MEP/dipole data test first-order response;
3. the frozen `3e-4`/`1e-3` Gate-A comparison bounds nonlinear contamination;
4. a positive/strongly-convex scalar, rather than the old MACE-POLAR fixed-point
   head, owns the new source and response.

It also warns that ordinary QEq can show incorrect dissociation and cubic
system-size polarizability.  The MAPLE head must therefore not collapse to an
unqualified global QEq patch.  Its response must use the richer two-width
radial source/field space and retain explicit passivity/root/domain gates.

## 2. Zhong et al., *Machine learning interatomic potential can infer electrical response*

- Preprint: https://arxiv.org/abs/2504.05169

The LES results show that long-range energy/force training can encode useful
electrical response without directly supervising atomic charges.  This is
consistent with using frozen MACE-MDP and zero-field MACE-POLAR latent features
as geometry descriptors.  It does **not** establish a quantitative molecular
nonuniform-field MEP, a PCM source, or a common solute-continuum functional, so
it is not evidence for reusing the original MACE-POLAR source head.

## 3. Zhang and Zavadlav, *ConSolv*

- Preprint: https://arxiv.org/abs/2606.24983

ConSolv is a new solvent-conditional implicit-solvent MLP spanning 66 organic
solvents.  It is relevant to the eventual solvent-descriptor interface, but it
combines ab-initio data with experimental solvation free energies.  MAPLE will
not use its total-solvation-target strategy to train or select the electronic
source/response head.  Solvent conditioning, if later adopted, must remain a
separate content-addressed continuum/nonpolar module and pass held-out solvent
transfer tests.

## 4. Thomas et al., *Self-consistent Coulomb interactions for MLIPs*

- Preprint: https://arxiv.org/abs/2406.10915

The mathematical locality analysis supports separating an explicit long-range
Coulomb functional from learnable local short-range terms while solving global
charge variables self-consistently.  It reinforces the current module boundary
between frozen geometry features, the scalar source functional, the analytic
source/receiver map, and the continuum functional.

## Frozen decision

No paper justifies changing the current targets to density-partition
coefficients, experimental solvation residuals, or stock MACE-POLAR source
imitation.  The newest MACE design-space paper instead strengthens the current
observable-first, Gaussian `l<=1`, scalar-functional direction.  The live dense
pilot provides repository-specific evidence beyond the literature: the two-
width `8N` span has about `1.47%` zero-field and `0.56%` response-MEP audit error
on the first molecule while retaining a negative response spectrum and a
positive Hessian-extension Gram matrix.

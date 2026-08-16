# MAPLE-CDS-W1 decision and validation contract

## Decision

The frozen MACE-MDP + MACE-POLAR hybrid electrostatic profile and stock
PySCF SMD-CDS remain unchanged until the complete v3 development result is
aggregated. The v3 records already retain the electrostatic polarization and
stock CDS leaves separately, so the following two baselines require no new
electronic or continuum calculation:

- **M0:** frozen hybrid electrostatics only;
- **M1:** frozen hybrid electrostatics plus stock SMD-CDS.

The SMD paper states that the electrostatic/non-electrostatic partition is
model-dependent and that SMD and SM8 CDS coefficients cannot be compared
unambiguously when their electrostatics differ (Marenich, Cramer, and
Truhlar, J. Phys. Chem. B 2009, DOI `10.1021/jp810292n`). Therefore a
profile-specific CDS is scientifically admissible, but only as a new model
identity.

One molecule for which stock CDS worsens the total error is a diagnostic, not
proof that the frozen electrostatic component is globally correct. The full
paired M0/M1 report decides whether stock-CDS transferability is a population
problem.

## Frozen boundaries

CDS development must not change:

- MACE-MDP or MACE-POLAR checkpoints;
- permanent point-multipole or induced Gaussian source definitions;
- receiver-field definition;
- dielectric, cavity radii, ddX equation, or solver tolerance;
- electrostatic energy ledger;
- geometry, standard state, or experimental record identity.

The fitted target is explicitly effective:

```text
y_effective = delta_G_experiment
              - G_hybrid_electrostatic
              - delta_G_standard_state
```

It can contain real cavity/dispersion/solvent-structure physics together with
residual errors from electrostatics, geometry, continuum approximations, and
experiment. It must not be described as an independently observed physical
component.

## Dataset contract

The frozen 505-record development panel has ten solvents:

- 306 water records;
- 199 records across nine nonaqueous solvents.

It has already been partially executed and cannot be repartitioned into a new
post hoc blind set. The existing frozen MNSol confirmation partition has 148
records, of which 102 do not share a solute geometry with the prior pilot. Its
water subset contains 81 records, 73 without prior-pilot geometry overlap.

Consequently:

- `MAPLE-CDS-W1` is a water-only candidate trained/selected only inside the
  306-record water development subset;
- its current MNSol confirmation is the frozen 81-record water subset, with
  overlap-stratified reporting;
- a claim requiring at least 150 untouched water molecules needs a separately
  sourced and preregistered external set;
- nonaqueous solvents require a distinct `MAPLE-CDS-MS1` or solvent-specific
  profile. W1 must never silently replace their CDS.

## Candidate ladder

The comparison is fixed before examining the complete v3 aggregate:

1. **M0:** zero CDS;
2. **M1:** stock SMD-CDS;
3. **M2:** one scalar multiplier on stock CDS, with no fitted intercept;
4. **M3:** a low-dimensional linear water SMD atomic-surface-tension basis.

M3 keeps SMD switching functions, exponents, radii, and chemical-environment
definitions fixed. Only preregistered linear coefficients may vary. A larger
local-environment model is considered only if grouped validation rejects M3.
Neural residual correction is outside W1.

The production area definition must be selected before fitting, content
addressed, differentiable, and identical in feature generation, energy, and
force evaluation. The legacy PySCF internal surface is not observable, and the
experimental Fibonacci grid is not a structural rotational guarantee; neither
may be silently substituted for the final surface.

## Fit and selection protocol

For a frozen design matrix `X`, fit only on development data using grouped,
nested validation. Groups must keep the same solute/geometry family together
and should additionally audit scaffold, homolog, functional-group, element,
and size leakage.

The primary candidate is regularized robust linear regression:

```text
argmin_theta sum_i w_i Huber(y_i - X_i theta)
            + lambda ||L(theta - theta_prior)||^2
```

The grids for Huber scale, regularization, prior strength, and model complexity
must be preregistered. No confirmation result may select them. Report design
rank, condition number, coefficient uncertainty, bootstrap intervals, and
held-group failure modes.

## Admission and stop rules

Report M0-M3 with MAE, RMSE, signed error, Q95 absolute error, maximum error,
per-chemistry strata, and bootstrap intervals. The already locked v3 decision
remains the preregistered point MAE threshold of `1.5 kcal/mol`; it cannot be
rewritten after execution. A future W1 confirmation should preregister the
stronger one-sided 95% MAE upper-confidence-bound threshold of
`1.5 kcal/mol`. RMSE, bias, Q95, and maximum error are tail-risk diagnostics;
hard numerical limits for them must be frozen before confirmation rather than
chosen after viewing it.

Stop this lane when any of the following holds:

- the frozen electrostatic/source diagnostics fail, so CDS would merely hide
  a source or continuum defect;
- grouped development validation does not improve over M0/M1 reproducibly;
- M3 is rank-deficient or requires unstable/large compensating coefficients;
- confirmation fails; no confirmation-set refit is permitted.

If W1 passes energy admission, force admission remains separate. Conservative
CDS forces require the analytic derivative of the exact fitted scalar,
including both area and environment-switch derivatives, followed by
translation/rotation covariance, multistep finite differences, closed-loop
work, distorted geometries, and downstream gates.

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

The linear tension algebra is already represented by one shared contract in
`implicit/smd_cds.py`:

```text
gamma_A(R; theta) = b_A(R) @ theta
X(R, A)_j         = sum_A A_A(R) b_Aj(R) / 1000
G_CDS(R; theta)   = X(R, A) @ theta
```

The 18 named columns contain ten elemental intercepts and the eight published
aqueous environment branches, including the published zero-coefficient H-O
branch. The stock coefficient vector reconstructs the existing SMD-water
atomic tensions, while the same basis supports an arbitrary frozen coefficient
vector and its analytic tension-coordinate VJP. The area vector remains an
explicit input: this algebra does not choose, fit, or silently substitute the
production surface definition.

The production area definition must be selected before fitting, content
addressed, differentiable, and identical in feature generation, energy, and
force evaluation. The legacy PySCF internal surface is not observable, and the
experimental Fibonacci grid is not a structural rotational guarantee; neither
may be silently substituted for the final surface.

### Smooth harmonic area candidate

The first coefficient-space candidate now has an explicit geometric parent.
The harmonic cavity field `e_i(R,u)` is constructed by applying a centered
smooth Heaviside directly to signed sphere overlap and composing those relaxed
exposure fractions. It is **not** constructed as the square root of a separate
area weight. Therefore its SMD-like geometric area is

```text
A_i(R) = a_i^2 integral e_i(R,u) dOmega
       = a_i^2 sqrt(4*pi) c_i,00(R).
```

The fact that electrostatic trial and test functions are both attenuated by
`e_i` does not force a local `e_i^2 dS` geometric measure: the two factors in
the electrostatic bilinear form occur at its trial and test arguments. Squaring
an exposure fraction would instead define a separately named quadratic
participation measure and would attenuate a transition point with `e=1/2` to
one quarter of its base area.

This decision is frozen from transition-profile provenance and geometry, not
from final solvation errors. It also agrees with the current PySCF 2.13.1 SWIG
surface semantics, whose exposed area is linear in its switching function
`swf`. `SmoothHarmonicExposureArea` evaluates the scalar from the shared
harmonic coefficient graph, rejects violations of `0 <= A_i <= 4*pi*a_i^2`
instead of clipping, and supplies its exact coordinate VJP. A CDS term may
claim the *same harmonic cavity* as an electrostatic continuum only after
`validate_same_cavity_as()` proves equality of radii, transition width,
exposure order, radial projection order, atom identity, and cavity profile.

This is a named smooth area model, not a claim of bitwise equality to sharp
SMD SASA. Its finite-width, finite-band geometry accuracy must be checked on
preregistered analytic union-of-spheres cases before any coefficient fit. The
full Pro-assisted mathematical audit and the local independent checks are
recorded in
`evidence/HARMONIC_CDS_AREA_PRO_AUDIT_2026-08-16.md`.

### Target-blind feature freeze

The first W1 design matrix is generated before any W1 fit by
`generate_maple_cds_w1_features.py` under an external, read-only
preregistration created by
`create_maple_cds_w1_feature_preregistration.py`. The generator parses the
licensed MNSol archive to recover and validate the already-frozen molecular
geometries. The loader necessarily parses the table's target column into its
in-memory record objects; the downstream feature computation never accesses
that attribute, a hybrid prediction record, or a confirmation-selection
manifest. Each exclusive record contains an opaque development identity,
geometry/configuration hashes, the 18-column design row, and a stock-coefficient
reconstruction control. It emits neither coordinates nor targets.

The geometry-only numerical choice is frozen as:

```text
SMD SASA radii:              published Bondi-style radii + 0.4 A probe
transition width:            0.18 A^2
exposure lmax:               4
radial quadrature order:     192
runtime:                     Torch float64 CPU
water development records:  306
```

Across those 306 frozen water geometries, the largest number of simultaneous
transition pair factors on one non-buried atom is 27. The exact finite-product
contraction therefore requires degree

```text
(27 + 1) * 4 = 112,
```

below the implementation limit of 128. This audit is recomputed from geometry
by both preregistration and feature generation; `lmax=4` was not selected from
solvation errors. The feature preregistration binds the parent hybrid-v3
preregistration, exact Git tree, every loaded repository source, four frozen
MNSol input files, water-identity fingerprint, area definition, basis ordering,
and stock control vector. Feature generation requires the same clean Git tree
and the same normalized Python/platform/package/NumPy/Torch runtime identity;
it writes only outside the checkout.

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

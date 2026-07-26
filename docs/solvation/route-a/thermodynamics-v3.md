# Route A protocol v3: one-measure soft-cutoff cluster QCT

Status: research contract. Runtime conditioning, packing analysis, finite
`n=0`/multi-`n` closure and the reviewed evidence bridges exist, but real
frozen-Hamiltonian replicas and blind-holdout gates are not complete.
Public `#solvfe` therefore remains fail-closed.

## Why v2 is diagnostic only

Protocol v2 used a hard occupancy indicator for the packing probability while
the fixed-`n` cluster calculation used a finite soft restraint.  Retaining that
restraint at both alchemical endpoints cancels an endpoint release term, but
does not make the restrained partition function equal to a hard-conditioned
partition function.  V3 removes this category mismatch by using one
complementary membership definition everywhere.

## One membership and its exact complement

For the signed nearest solute atom-sphere distance of water oxygen `j`,

```text
z_j       = (d_s(O_j) - lambda_s) / R
b_j       = 1 / (1 + exp(z_j))
1 - b_j   = 1 / (1 + exp(-z_j))
u_member  = -kT ln b_j
u_empty   = -kT ln(1 - b_j)
```

Thus `exp(-beta u_member)=b` and `exp(-beta u_empty)=1-b` exactly.
`lambda_s`, `R`, the solute atom map, radii, solute measure, temperature and
boundary conditions are hash-bound.  The hard-cutoff limit is a required
sensitivity test; a smoother field is not assumed to be more accurate.

The soft integer-occupancy weights are the coefficients of

```text
product_j [(1-b_j) + t b_j].
```

They are evaluated by an `O(N^2)` recurrence and must be non-negative and sum
to one on every frame. Production analysis preregisters active occupancies
`0..nmax`, aggregates every larger occupancy into one tail bin, and
preregisters a well-conditioned simplex reconstruction coordinate. It never
reconstructs the numerical `n=Nwater` tail merely because it is last. Bulk-box
regressions cover 64, 128, 256 and 512 waters. In particular,

```text
omega_0 = product_j (1-b_j).
```

Production packing and occupancy artifacts require `nmax >= 1`, so their
active support always contains both `n=0` and `n=1`; a packing-only
`nmax=0` artifact cannot be bridged into the QCT ledger. For finite systems,
`nmax` may equal the total number of explicit solvent molecules. That is a
valid full-support representation with an exactly zero aggregate tail, not an
out-of-range request.

## Packing identity

The packing ensemble is the product of a declared solute measure and a
periodic pure-water measure.  There are no physical solute-water
interactions.  The full complementary field is

```text
U_empty = sum_j [-kT ln(1-b_j)].
```

The two exactly equivalent definitions are

```text
p0_soft = <exp(-beta U_empty)>_0
        = <product_j (1-b_j)>_0

-RT ln p0_soft = F(scale=1) - F(scale=0).
```

The primary result is the MBAR free-energy difference.  Reweighting the soft
empty weight to the explicit scale-zero target state is mandatory as an
equality check. A single `PackingOccupancyBridge` additionally proves that
this result and reference `p_tilde(n)` use the identical reduced-potential
table, frame order, `omega_0`, target row, membership, solute measure, water
Hamiltonian and uncertainty gate. An arbitrary replacement `p0` cannot enter
the ledger. `RT` is derived from the scheduled temperature, never accepted as
a caller-selected conversion. The schedule stores explicit target and full-field indices;
neither the first row nor the last row has an implicit scientific meaning.
A hard empty-shell count may be reported only as a diagnostic.

The two packing estimators are correlated because they use the same reduced-
potential table. Their agreement uncertainty is therefore obtained from a
state-stratified paired bootstrap that recomputes both estimators in every
replicate. Adding their marginal variances as if the estimators were
independent is forbidden. The bootstrap seed, replicate count, paired-value
hash, covariance and direct difference uncertainties are part of the sealed
packing result.

The implemented packing ensemble is NVT with a fixed, hash-bound periodic
cell and `pressure_bar=null`. An NPT label is rejected until a variable-cell
runtime supplies stress, the `PV` term and the corresponding volume measure;
the protocol's 1 bar thermodynamic reference does not by itself turn a
fixed-cell trajectory into NPT sampling.

The pure-water Hamiltonian is not yet frozen.  Before a real v3 campaign it
must be an identified, hash-bound checkpoint with verified periodic
bulk-water support.  MACE-OFF24 or MACE-OMOL cannot enter this row silently.

## Fixed-`n` association

Each labeled cluster water uses `u_member=-kT ln b`.  The same member field
remains at both alchemical endpoints.  The decoupled translational integral
for one labeled water is therefore

```text
V_eff = integral b[d_s(r)] d^3r.
```

Calling the release contribution zero is valid only relative to this declared
soft-membership target.  It is not a conversion to hard occupancy.

The coupled and uncoupled explicit-solvent rows must use the same membership
hash and solute measure as packing:

```text
p_tilde(n) = <omega_n>_0
x_tilde(n) = <omega_n>_X
p0_soft    = p_tilde(0)
x0_soft    = x_tilde(0).
```

Here the solute-measure hash identifies the coordinate measure, constraints,
registered conformer support and atom map.  It does not assert that the
reference and coupled Hamiltonians generate the same conformer probability
distribution.

## One v3 master equation

For the same periodic reference and coupled Hamiltonians, define

```text
Z_0,n = sum_q omega_n(q) exp[-beta U_0(q)]
Z_X,n = sum_q omega_n(q) exp[-beta U_X(q)]
g_n   = -RT ln[Z_X,n/Z_0,n].
```

The exact conditioned row is

```text
A_n_exact = g_n - RT ln[p_tilde(n)/p0_soft].
```

The exact full-support multi-occupancy master equation is

```text
mu_A = -RT ln(p0_soft)
       -RT ln sum_n exp[-beta A_n].
```

For preregistered active support `0..nmax` with aggregate
`x_tail=sum_(n>nmax) x_tilde(n)`, the runtime uses the algebraically equivalent
active-support form

```text
mu_A = -RT ln(p0_soft)
       -RT ln sum_(n=0)^nmax exp[-beta A_n]
       +RT ln(1-x_tail).
```

Both `p_tail` and `x_tail` must pass the preregistered tail-mass gate; the
free-energy tail envelope remains a separate promotion gate.

Its fixed-occupancy and direct-empty-shell cross-checks are

```text
mu_n      = A_n + RT ln[x_tilde(n)/p0_soft]
mu_direct = A_0 + RT ln[x0_soft/p0_soft].
```

These relations also reconstruct the coupled distribution:

```text
x_reconstructed(n) =
  p0_soft exp{beta[mu_A-A_n]}.
```

`p_tilde(n>0)` and `x_tilde(n)` are therefore not second additive ledger
terms. A separate, covariance-bearing `g_n` profile makes every active
`p_tilde(n)` enter the periodic-Hamiltonian bridge/falsifier;
`x_tilde(n)` supplies the fixed-`n` closure test.  Adding either distribution
again on top of density, `V_eff` and `n!` would double count occupancy
statistics.

## Labeled cluster construction and approximation boundary

MAPLE constructs an approximate nonperiodic cluster-continuum row once:

```text
A_n_cluster =
  sum_i DeltaG_alch_labeled(i)
  + sum_i[-RT ln(rho_W_number_per_A3 V_eff,i)]
  + RT ln(n!)
  + sum_i DeltaG_release(i)
  + DeltaG_outer(XW_n)
  - n DeltaG_outer(W).
```

The density-volume factor occurs once per edge and `RT ln(n!)` occurs once per
row when converting sequential labeled waters to the unlabeled occupancy
convention.  No `p_tilde(n)` factor is then added to this row.

For `n=0`, all edge, density-volume and symmetry sums are empty:

```text
A_0 = DeltaG_outer(X | soft-empty-conditioned ensemble).
```

The nonperiodic cluster row and the periodic exact row have different
Hamiltonians and boundary conditions. They share one boundary-independent
membership-surface identity but use distinct, hash-bound minimum-image and
nonperiodic boundary adapters. Every `V_eff` is a real
`SoftEffectiveVolumeEstimate` artifact bound into its cluster row. The rows
are connected only through explicit boundary-measure and Hamiltonian bridges.
Their difference is a preregistered
cluster-continuum embedding residual and falsification gate, not an exact
identity.  The current monomer water reference remains a research hypothesis
and must later be compared with the frozen solvent-cluster reference
ablation.

## Covariance and finite closure

The ledger propagates one joint covariance over

```text
[p_tilde(0..nmax), p_tail,
 x_tilde(0..nmax), x_tail,
 A_0..A_nmax, g_0..g_nmax].
```

The `p` and `x` blocks retain their probability-simplex nullspaces; the matrix
is never inverted.  Production uncertainty must come from a hash-bound joint
provenance-graph bootstrap so shared water references, alchemical edges,
outer rows and data-dependent tail selection remain correlated. The shared
case requires replicate-level `QCTBootstrapEvidence`; a naked evidence hash is
rejected. A block-diagonal matrix is allowed only with explicit independence
evidence and must have exactly zero cross-block entries.

Shared-bootstrap columns use one canonical labeled ledger order and one source
artifact binding across the reference distribution, coupled distribution,
conditioned profile and conditional-coupling profile. A positive propagation
gate must retain at least one nonzero cross-block covariance and demonstrate
that it changes the direct-`n=0` and multi-`n` standard errors relative to the
block-diagonal counterfactual. A zero-cross-covariance-only test cannot certify
the shared path.

The finite enumerator now proves, with fractional `omega_n`, that direct
`n=0`, every fixed-`n` row and the multi-`n` log-sum are identical for one
exact Hamiltonian.  A second enumerable labeled-cluster construction includes
primitive coupling factors, `rho*V_eff`, release factors, outer factors and
division by `n!` exactly once. It is independent of the additive cluster
builder. Deliberately applying the actual density-volume or factorial
correction twice is required to fail closure.

Exact-enumeration labels are not caller-selected metadata. Exact occupancy,
conditional-coupling and zero-covariance artifacts are emitted only by the
internal finite-system oracle and share one source hash. Empirical artifacts
cannot enter an exact ledger by replacing an estimator string or supplying an
arbitrary evidence hash.

The finite periodic boundary adapter is likewise canonical rather than a
literal placeholder. Its identity is derived from the membership definition,
observation volume, solute measure, periodic boundary declaration and finite
coordinate-adapter contract, and is included in both the finite source hash
and Hamiltonian bridge hash.

These runtime and finite-enumeration closures do not promote the fixed-`n=1`
acetone value.  Real `n=0`, multi-`n`, periodic occupancy and cluster rows have
not yet been sampled under a frozen bulk-water Hamiltonian.

## Outer supermolecule continuum

Every `XW_n` frame uses one whole `solute + n waters` atom-centered GePol
cavity.  Vacuum and continuum endpoints have the same atoms, coordinates,
conditioning measure and cavity construction.  PCM is evaluated per frame,
energy-only.  Native or PEDRA warnings are fatal and frame deletion is
forbidden.  CDS remains outside the v3 core until a separately hash-bound
development ablation is frozen.  The outer provider must not hide a packing
term.

## Promotion gates

The research workflow remains fail-closed until all of the following pass:

1. analytic complementarity, force and occupancy-normalization tests;
2. packing MBAR overlap, ESS, BAR--MBAR, uncertainty, time-stability and
   estimator-equivalence gates;
3. real-ensemble `n=0`, multi-`n`, `p0/x0/p(n)/x(n)` and joint-bootstrap
   covariance closure;
4. three independent v3 replicas;
5. whole-supermolecule outer equivalence with zero warnings and no deletions;
6. a frozen development protocol followed by a blind multi-molecule holdout;
7. prespecified paired error lower than the frozen Route 2 baseline.

The result remains a cluster-continuum approximation to molecular QCT and is
not labeled exact.

## Primary literature

- S. Chempath, L. R. Pratt and M. E. Paulaitis, *J. Chem. Phys.* **130**,
  054113 (2009), <https://doi.org/10.1063/1.3072666>.
- D. N. Asthagiri, M. E. Paulaitis and L. R. Pratt, *J. Phys. Chem. B*
  **125**, 8298-8314 (2021),
  <https://doi.org/10.1021/acs.jpcb.1c04182>.
- D. Sabo et al., *J. Phys. Chem. B* **112**, 867-876 (2008),
  <https://doi.org/10.1021/jp075459v>.
- B. Xi et al., *J. Chem. Theory Comput.* **20**, 4219-4228 (2024),
  <https://doi.org/10.1021/acs.jctc.4c00116>.
- M. R. Shirts and J. D. Chodera, *J. Chem. Phys.* **129**, 124105 (2008),
  <https://doi.org/10.1063/1.2978177>.

# Route A protocol v2: conditioned cluster-QCT supermolecule cycle

## 1. Why v2 exists

The real acetone + one-water pilot falsified two assumptions in protocol v1:

1. the atom-centered Route 2 cavity changed topology across an ordinary
   interacting endpoint ensemble; and
2. the ordinary bare-solute Route 2 value is not the `n=0` QCT state.

Protocol v1 used `G_0 = ΔG_outer(X)` and then formed a raw multi-`n` log-sum.
That shortcut omits the probability of forming an empty inner-shell
observation volume and does not condition the `n=0` long-range state on that
empty volume.  V1 remains a reproducible historical engineering protocol, but
its multi-`n` result is not eligible for scientific promotion.

Protocol v2 implements a cluster-QCT approximation whose missing terms,
conditioning measure, and outer-cavity geometry are explicit.  It is a
research protocol until all statistical, cavity, ablation, and blind-holdout
gates pass.

## 2. State definitions

`X` is one neutral, closed-shell solute and `W` is water.  The inner-shell
indicator uses oxygen membership:

`n(q,lambda_s) = sum_j 1[d_s(O_j) <= lambda_s]`.

`d_s` is the all-solute-atom signed-distance map.  Once an oxygen is inside,
the complete water molecule is retained.  The observation volume is the
oxygen-center volume defined by `lambda_s`; it is not the PCM cavity.

The following probabilities are distinct and must never be substituted for
one another:

- `p_0(lambda_s)`: probability that the observation volume is empty in pure
  bulk water, without the solute--water interaction;
- `x_0(lambda_s)`: probability that the solute inner shell is empty in the
  fully interacting solution;
- `p_X(n;lambda_s)`: probability of occupancy `n` around the interacting
  solute.

V2 uses the cluster-QCT multi-state form and therefore requires
`p_0(lambda_s)` explicitly.  `x_0` and `p_X(n)` remain required cross-checks
when a direct-QCT or fixed-`n` identity is evaluated.

## 3. Cluster-QCT ledger

The production target is

`mu_X_ex = -RT ln p_0(lambda_s) - RT ln sum_n exp[-beta A_n_tilde(lambda_s)]`.

`A_n_tilde` is the negative logarithm of the conditioned equilibrium weight:

`A_n_tilde(lambda_s) = -RT ln[K_n_tilde(lambda_s) rho_W^n]`.

MAPLE estimates it through sequential labeled-water insertion:

`A_n_tilde = sum_{i=1..n}[DeltaG_alch_labeled(i) + DeltaG_vol(i) + RT ln(i) + DeltaG_release(i)] - n RT ln(rho_W_over_C0) + DeltaG_LR_common(XW_n | n,lambda_s) - n DeltaG_outer(W)`.

For the v1 restraint contract,

`DeltaG_release(i)=0`

because the same flat-bottom shell restraint remains in the two interacting
endpoints and no placement or orientation restraint is introduced.  A future
restraint changes this term and requires a new contract.

The volume-density identity remains

`DeltaG_vol(i) - RT ln(rho_W_over_C0) = -RT ln[rho_W_number_per_A3 V_eff,i]`.

The `n=0` entry is

`A_0_tilde = DeltaG_LR_super(X | n=0,lambda_s)`.

It is evaluated on the empty-shell conditioned ensemble with the same
preregistered construction rule used for every other `n`.  The actual cavity
depends on the atoms in the `XW_n` supermolecule, as required by the
`mu_XWn_ex - n mu_W_ex` QCT term.  An unconditioned single-point or ensemble
Route 2 bare-solute hydration free energy cannot replace it.

The direct-QCT identity

`beta mu_X_ex = ln x_0(lambda_s) - ln p_0(lambda_s) + beta mu_LR(lambda_s)`

and the fixed-`n` identity

`mu_X_ex = -RT ln[K_n^(0) rho_W^n] + RT ln p_X(n;lambda_s) + mu_XWn_ex - n mu_W_ex`

are mandatory thermodynamic cross-checks on a development subset.  Agreement
is a falsification gate, not a license to fit terms to experiment.

## 4. Whole-supermolecule outer cavity

Every `XW_n` frame is one supermolecule.  The continuum boundary is generated
from the complete atom list for that frame.  Solute and explicit waters are
not evaluated in separate cavities.  All occupancies and the water reference
use the same preregistered construction rule: the frozen base radii, radius
scale, tessera area, added-sphere threshold, dielectric model, and warning
policy are identical.

The mesh is regenerated per frame because the explicit water is mobile.  A
single restart mesh large enough to contain every allowed water position was
tested and rejected: it was numerically stable but moved the boundary far from
the physical supermolecule and destroyed the cluster/water outer-cycle error
cancellation.  Per-frame GePol is therefore an energy-only postprocessing
operator; v2 does not claim continuum forces.

Before any PCM energy is accepted:

1. the scaled whole-supermolecule sphere union has exactly one component;
2. every solute--water bridge exceeds the frozen overlap margin;
3. the native mesh has one component and no protected-sphere penetration;
4. every sampled frame evaluates with zero native or PEDRA warnings; and
5. the atom list, restraint, measure, boundary conditions, cavity-rule hash, and
   coordinates match between each vacuum/outer endpoint pair.

Deleting failed frames is forbidden.  The full ensemble either passes or the
outer free energy is undefined.

The v2 development candidate uses a literature-derived `1.2` scale on the
frozen SMD/Bondi-like Coulomb radii, `AREA=0.20 A^2`, and
`MINRADIUS=1.00 A`.  The scale was selected from PCM practice, not from the
acetone experimental value.  The tessera area was selected by warning-free
geometry and numerical-convergence probes: the representative outer-cycle
electrostatic energy changed by about `0.006 kcal/mol` between
`AREA=0.15 A^2` and `0.20 A^2`.  These values remain development candidates
until a multi-molecule geometry panel passes.

The legacy unscaled Route 2 cavity remains a diagnostic only.  A warning-free
PCM result does not rescue a disconnected or near-tangent sphere union.

## 5. Electrostatics, packing, and nonpolar terms

The v2 core outer term contains the occupancy-conditioned long-range
electrostatic response on one whole-supermolecule cavity per frame:

`DeltaG_LR_super(XW_n) = F[U_OMOL_vac + delta_PCM_super,R_n] - F[U_OMOL_vac,R_n]`.

`U_OMOL_vac` is counted once.  The PCM half-coupling is counted once.  The
MACE-POLAR intrinsic response may enter only through the frozen delta
Hamiltonian; a complete Route 2 solvated total energy may not be added.

`-RT ln p_0(lambda_s)` is the inner-volume packing term.  It is not hidden in
the outer provider.

SMD CDS is excluded from the v2 core because v2 changes the electrostatic
radii and because CDS already represents short-range solvent-structure physics
that the explicit first shell is intended to supply.  Reusing the legacy CDS
term unchanged would mix parameterizations and risk double counting.  A
hash-bound development ablation must compare:

- no CDS in the v2 core;
- one solute-only legacy CDS term counted once; and
- a separately justified supermolecule nonpolar model.

The final choice is frozen on development molecules before the blind holdout.

## 6. Sampling and probability estimators

`p_0(lambda_s)` is obtained from a pure-water occupancy calculation at the
same temperature, pressure, water model / Hamiltonian, signed-distance
observation-volume definition, and `lambda_s`.  Rare empty-volume sampling
uses staged soft-cavity bias and an overlap-connected free-energy estimator;
counting zero visits in an unbiased trajectory is not an estimator.
The maximum-bias state must contain at least `100` empty-volume frames, and
the MBAR-reweighted packing free energy must have standard error no larger
than `0.2 kcal/mol`.

Each insertion edge and conditioned outer edge requires independent replicas,
neighboring-state overlap, BAR--MBAR agreement, effective sample size, time
stability, and exact state hashes.  Periodic MACE-OFF24-SC trajectories may
propose structures and may support a separately declared occupancy
sensitivity study, but they cannot be silently reweighted into the
nonperiodic OMOL target measure.

Before sampling, v2 freezes at least three independent replicas, a maximum
adjacent-state standard error of `0.2 kcal/mol`, and pairwise replica
agreement within `2.0` combined standard errors.  Each replica must pass its
own overlap, ESS, BAR--MBAR, and half-trajectory gates before any pooled
estimate is formed; pooling cannot conceal a failed replica.

## 7. Scientific promotion boundary

V2 can be called scientifically validated only after:

- complete, warning-free common-cavity evaluation for every retained frame;
- converged `p_0`, insertion, and conditioned outer estimates;
- direct-QCT / fixed-`n` cycle cross-checks on the mandatory subset;
- development-set selection of shell, CDS/nonpolar, and reference-cycle
  choices without looking at the holdout;
- a frozen blind holdout with uncertainty intervals and paired errors; and
- lower prespecified primary error than frozen Route 2.

Until then, output is diagnostic and must not claim that Route A is more
accurate than Route 2.

## 8. Primary sources behind the contract

- Pratt et al., quasi-chemical organization of hydration:
  <https://arxiv.org/abs/physics/9909004>
- Rogers and Beck, exact direct-QCT decomposition:
  <https://arxiv.org/abs/0809.1628>,
  DOI <https://doi.org/10.1063/1.2985613>
- Merchant and Asthagiri, conditioned multi-state / molecular-QCT analysis:
  <https://arxiv.org/abs/0910.1786>
- Bryantsev, Diallo, and Goddard, cluster-continuum thermodynamic cycles:
  <https://doi.org/10.1021/jp802665d>
- PCMSolver cavity and restart input:
  <https://pcmsolver.readthedocs.io/en/v1.1.3/users/input.html>

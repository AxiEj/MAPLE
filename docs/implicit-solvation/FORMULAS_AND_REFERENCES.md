# Formula and reference ledger

This file is the human-review ledger for MAPLE's Route 1 fixed-charge PB/GB
implementation. Equations implemented directly in MAPLE are listed explicitly;
upstream continuum and charge-generation algorithms are called through their
documented provider interfaces rather than reimplemented from memory.

## Fixed-charge composition

MAPLE adds only the solvent correction to the MLIP gas potential:

\[
E_{\mathrm{solution}}(R)=E_{\mathrm{MLIP,gas}}(R)
 +G_{\mathrm{polar}}(R,q_{\mathrm{fixed}})
 +G_{\mathrm{nonpolar}}(R).
\]

No OpenMM, APBS, or Amber gas-phase molecular-mechanics energy is added.

## Complete composed numerical Hessian

For a force-capable fixed-charge provider, the reported force is

\[
F(R)=-\nabla_R E_{\mathrm{solution}}(R)
=F_{\mathrm{MLIP,gas}}(R)
+F_{\mathrm{polar}}(R,q_{\mathrm{fixed}})
+F_{\mathrm{nonpolar}}(R).
\]

The implicit-solvent numerical Hessian is the derivative of this **complete**
force, not the gas-backend Hessian:

\[
H_{ij}(R)=\frac{\partial^2E_{\mathrm{solution}}}
{\partial R_i\partial R_j}
=-\frac{\partial F_i}{\partial R_j},
\]

\[
H_{ij}^{\mathrm{FD}}(R)=
-\frac{F_i(R+\delta e_j)-F_i(R-\delta e_j)}{2\delta}
+\mathcal{O}(\delta^2).
\]

MAPLE symmetrizes the finite-difference matrix and keeps the fixed charges and
provider parameters unchanged at every displacement. The calculation requires
two complete force calls for each movable Cartesian coordinate. It is
therefore available only for force-capable OpenMM GB compositions and only
after explicit `hessian=numerical` selection. Using an analytic gas-MLIP
Hessian would omit
\(\partial^2(G_{\mathrm{polar}}+G_{\mathrm{nonpolar}})/\partial R^2\)
and is rejected.

This Hessian supplies local vibrational curvature of the Route 1 potential. It
does not supply a gas/solution partition-function difference, conformer
ensemble, solution standard-state conversion, or absolute solvation free
energy. MAPLE's existing FREQ translational and rotational RRHO terms retain
their ideal-gas pressure convention and are explicitly labeled as such when an
implicit correction is attached.

### Low-frequency RRHO/qRRHO formulas

For a real vibrational wavenumber \(\tilde\nu_i\), MAPLE uses the
Chai--Head-Gordon switching function

\[
w_i=\frac{1}{1+(\nu_0/\tilde\nu_i)^4}.
\]

The free-rotor entropy is evaluated with a finite effective inertia,

\[
\mu_i=\frac{h}{8\pi^2 c\tilde\nu_i},\qquad
\mu_i'=\frac{\mu_i B_{\mathrm{av}}}{\mu_i+B_{\mathrm{av}}},
\]

\[
S_{\mathrm{FR},i}
=R\left[\frac12+
\ln\left(\frac{8\pi^3\mu_i'k_BT}{h^2}\right)^{1/2}\right],
\]

where \(B_{\mathrm{av}}\) is the mean molecular moment of inertia. Thus
`ilowfreq=2` uses

\[
S_i=w_iS_{\mathrm{HO},i}+(1-w_i)S_{\mathrm{FR},i}
\]

while retaining the harmonic vibrational internal energy. `ilowfreq=3` uses
the same entropy and additionally applies the Otlyotov--Minenkov interpolation

\[
U_i=w_i h c\tilde\nu_i
\left[\frac12+
\frac{1}{\exp(hc\tilde\nu_i/k_BT)-1}\right]
+(1-w_i)\frac{RT}{2}.
\]

The harmonic zero-point term is already inside the first term of this
expression; adding an undamped ZPE again would double count the low-frequency
limit. The default \(\nu_0=100\ \mathrm{cm^{-1}}\) matches MAPLE's historical
surface and a common qRRHO default. In the preregistered Route 1 v6 pilot,
however, local harmonic RRHO (`ilowfreq=0`) is primary. Grimme and
Otlyotov--Minenkov treatments at 50, 100, and 150
\(\mathrm{cm^{-1}}\) are sensitivity-only because summing explicit minima and
also assigning free-rotor entropy to the same torsional coordinate can double
count configurational entropy.

The v6 pilot stopped label-blindly on its first ANI2x/nitromethane record:
both phase optimizations converged, but the frozen absolute raw-Hessian
asymmetry gate failed by over three orders of magnitude. Symmetrized
multi-displacement spectra were stable in a post-failure diagnostic, which
motivates a separate numerical-qualification design but does not alter,
rescue, or rescore the v6 result.

The separately frozen v8 successor does not reinterpret that failure. A
three-MLIP, two-phase, three-displacement label-blind qualification replaces
the universal absolute acceptance threshold with a broad
`0.02 Hartree/Angstrom^2` sanity ceiling plus a scale-aware relative
Frobenius-asymmetry gate of `0.002`; the independent selected-frequency RMS
limit remains `25 cm-1`. Force-converged structures with a selected negative
mode are not declared minima: v8 displaces both signs of the most negative
mode by at most `0.5 Angstrom` per atom, reoptimizes the unchanged phase
potential, accepts only the lower converged structure with at least
`0.001 kcal/mol` energy lowering, and reapplies every Hessian and
stationary-point gate.

V8 later failed closed on the flexible preflight before reaching those Hessian
tests: one frozen AIMNet2 gas branch remained above the force threshold after
500 LBFGS steps. A separate label-blind optimizer-robustness qualification
then tested BFGSLineSearch, LBFGSLineSearch, and FIRE2/ABC on the identical
three MLIPs, two physical phases, and three selected source states. The
candidate changes only the numerical minimizer; every branch still
differentiates either
\(E_{\mathrm{MLIP,gas}}\) or
\(E_{\mathrm{MLIP,gas}}+G_{\mathrm{polar}}+G_{\mathrm{nonpolar}}\).
All three candidates passed only `12/18` branches under the common
force, finite-trace, non-increasing-energy, step, and calculator-evaluation
gates. Because the protocol forbids per-model or per-phase selection, no
global policy exists and no v9 thermochemistry protocol is scientifically
admissible from this screen.

The entropy interpolation follows Grimme,
DOI `10.1002/chem.201200497`; conformer-ensemble and msRRHO context follows
Pracht and Grimme, DOI `10.1039/D1SC00621E`; and the energy interpolation
follows Otlyotov and Minenkov, DOI `10.1002/jcc.27129`. These gas-phase
statistical-mechanics references establish the formulas only. They do not
validate their use as a solution free energy or establish a Route 1 accuracy
gain.

## Prebuilt explicit-inner / implicit-outer composition

For a supplied solute-plus-inner-solvent cluster, Route 1 applies the same
fixed-charge composition to the entire cluster:

\[
E_{\mathrm{cluster,outer}}(R)=
E_{\mathrm{MLIP,gas}}(\mathrm{cluster};R)
+G_{\mathrm{polar,outer}}(R,q_{\mathrm{fixed}})
+G_{\mathrm{nonpolar,outer}}(R).
\]

This is the energy differentiated by GB SP/OPT/SCAN. It is a fixed-shell
cluster-continuum configurational potential, not an absolute solvation free
energy. Here fixed shell means fixed component membership and coordination
number, not fixed coordinates.

For comparison, the consistent cluster cycle written at the 1 M standard
state is

\[
\Delta G_{\mathrm{solv}}^*(A)=
\Delta G_{\mathrm{clust,g}}^*(A(\mathrm{H_2O})_n)
+\Delta G_{\mathrm{solv}}^*(A(\mathrm{H_2O})_n)
-\Delta G_{\mathrm{solv}}^*((\mathrm{H_2O})_n)
-RT\ln([\mathrm{H_2O}]/n).
\]

The prebuilt runtime supplies only the cluster gas potential and outer
continuum contribution needed inside such a construction. It does not supply
the gas cluster-formation or occupancy free energy, solvent-cluster reference
or equivalent solvent chemical potential, standard-state conversion, or
ensemble over cluster isomers and coordination numbers.

- Bryantsev, Diallo, and Goddard, “Calculation of Solvation Free Energies of
  Charged Solutes Using Mixed Cluster/Continuum Models,” *J. Phys. Chem. B*
  (2008), DOI `10.1021/jp802665d`.
- da Silva, Svendsen, and Merz, “Explicitly Representing the Solvation Shell in
  Continuum Solvent Calculations,” *J. Phys. Chem. A* (2009),
  DOI `10.1021/jp809712y`,
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC2700946/>.
- Tomanik et al., “Solvation Energies of Ions with Ensemble
  Cluster-Continuum Approach,” *Phys. Chem. Chem. Phys.* (2020),
  DOI `10.1039/D0CP02768E`.

An alternative supermolecular cycle, used by CREST Quantum Cluster Growth
(QCG), evaluates

\[
\Delta G_{\mathrm{solv}}^*
=G^*[A(S)_n]-G^*[(S)_n]-G^*[A]
+\Delta G_{\mathrm{standard\ state}}.
\]

QCG generates both the solute-solvent and reference solvent ensembles,
includes conformational and thermochemical terms, and explicitly reports its
gas/solution reference states. This is a complete free-energy scaffold rather
than a fixed-shell potential. It is not currently a generic Route 1 backend,
however. The documented QCG grow, ensemble, and frequency levels are from the
GFN-xTB/GFN-FF family. At audited CREST commit
`cfdc301f759686b0fd66ced63b5ddbd6c693fa4f`, QCG constructs direct xTB
single-point, optimization, and `--hess`/`--ohess` commands. CREST's separate
`generic` energy/gradient subprocess is not called anywhere in the QCG source.
Thus CREST's generic backend is not wired into QCG, and ordinary CREST
generic-backend support is not evidence of an arbitrary-MLIP QCG free-energy
path.

FEBISS is an optional proposal preprocessor: it analyzes a supplied explicit
solvent trajectory with GIST, ranks solvent sites, and writes a selected
microsolvated structure. Its site free energies do not replace the pure-solvent
reference, standard-state, conformer/coordination ensemble, or target-MLIP
free-energy terms in either cycle above.

- Spicher et al., “Automated Molecular Cluster Growing for Explicit Solvation
  by Efficient Force Field and Tight Binding Methods,” *J. Chem. Theory
  Comput.* (2022), DOI `10.1021/acs.jctc.2c00239`.
- Kaur et al., “Free Energy Based Identification of Solvation Sites,”
  *Molecules* (2021), DOI `10.3390/molecules26061793`.
- Lehmann, Jameel, and Kaupp, “Systematic Evaluation of a
  Cluster-Continuum-Model Workflow to Compute the Free Energies of Solvation
  of Ions in Different Solvents,” *J. Phys. Chem. A* (2026),
  DOI `10.1021/acs.jpca.6c00886`.
- Frozen implementation and admission audit:
  [`route1-explicit-inner-implicit-outer-feasibility-audit-2026-07-25.json`](benchmarks/route1-explicit-inner-implicit-outer-feasibility-audit-2026-07-25.json).

## Discrete MLIP conformer reweighting diagnostic

For the development-only discrete-conformer diagnostic, MAPLE evaluates the
same set of conformer states in gas and implicit solution:

\[
\Delta G_{\mathrm{hyd}}^{\mathrm{disc}}
=-RT\ln
\frac{\sum_i \exp[-\beta(\Delta U_i^{\mathrm{MLIP}}+W_i)]}
{\sum_i \exp[-\beta\Delta U_i^{\mathrm{MLIP}}},
\qquad
W_i=\Delta G_{\mathrm{polar},i}+\Delta G_{\mathrm{nonpolar},i}.
\]

Only relative MLIP energies enter, so any additive MLIP energy zero cancels.
The reusable
`maple.function.free_energy.analyze_discrete_conformer_ensemble` core accepts
gas-energy and fixed-charge-solvent-correction arrays in `kcal/mol`; it has no
MLIP-specific import, checkpoint, conformer generator, or experimental-label
input. Any registered gas-phase MLIP can therefore provide the \(U_i\) values
on a common state set.

The core reports gas and solution weights, effective conformer counts,
dominant-state weights, distribution overlap, total-variation distance, the
Bhattacharyya coefficient, endpoint bounds, and a partition-ratio/FEP identity
check. Its default diagnostic gate requires at least two states, gas and
solution effective counts of at least two, no dominant weight above `0.95`,
distribution overlap of at least `0.10`, and the endpoint bound. Passing those
checks detects obvious concentration or support failures; it cannot prove that
the submitted conformers cover either continuous ensemble.

The sealed
[`route1-discrete-conformer-core-replay-2026-07-25.json`](benchmarks/route1-discrete-conformer-core-replay-2026-07-25.json)
replays 40 historical MACE-OFF23m records comprising 2,596 states. It
reproduces all 11 common legacy fields exactly. The AM1-BCC/OBC-II and
ABCG2/OBC-II sets each pass the weight diagnostic on only 14 of 20 cases.
Historical source records contain experimental fields, but neither the core nor
the replay metrics read them. This is MACE-only replay evidence for an
MLIP-agnostic API, not a named multi-MLIP validation.

A separate prospective common-domain diagnostic now evaluates the same 1,270
states from 19 cases with MACE-OFF23m, AIMNet2, and ANI2x. The fixed
AM1-BCC/OBC-II/ACE \(W_i\) array is identical for every model; only the
gas-phase relative MLIP energies change. Checkpoint-derived element tables are
intersected before evaluation, excluding the one phosphorus case because
ANI2x does not support P. The label-free energy artifact is
[`route1-multi-mlip-discrete-conformer-2026-07-25.json`](benchmarks/route1-multi-mlip-discrete-conformer-2026-07-25.json).

The final discrete correction is usually less model-sensitive than the
underlying weights: the across-model correction range has median
`0.03597 kcal/mol` and p90 `0.11969 kcal/mol`, but reaches
`1.41020 kcal/mol` for alachlor. MACE-OFF23m, AIMNet2, and ANI2x pass the
weight diagnostic on only `13/19`, `4/19`, and `12/19` cases, respectively.
The AIMNet2/ANI2x median gas and solution weight overlaps are only `0.2141`
and `0.2013`. Thus similar partition ratios can conceal different conformer
populations; the weight diagnostics cannot be replaced by agreement of the
single final number.

The prospectively frozen `1e-6 kcal/mol` repeat-energy gate remains failed:
float32 CUDA repeats differ by at most `7.59e-5 kcal/mol` in relative state
energy and `3.93e-6 kcal/mol` in the resulting discrete correction. The
threshold was not relaxed after observing the run. Historical MACE relative
energies reproduce within `7.14e-10 kcal/mol`.

Only after the label-free artifact was sealed was the development score opened.
The common fixed-geometry MAE is `1.930 kcal/mol`; discrete weighting gives
`1.910`, `1.969`, and `1.993 kcal/mol` for AIMNet2, MACE-OFF23m, and ANI2x.
Every paired bootstrap interval for MAE gain crosses zero. No gas MLIP is
selected or promoted from that label-exposed comparison.

Batching changes only how the \(U_i^{\mathrm{MLIP}}\) values are evaluated:

\[
\{U_i^{\mathrm{MLIP}}\}_{i=1}^{N}
=\operatorname{calculate\_many}(\{R_i\}_{i=1}^{N}).
\]

It introduces no new term into either numerator or denominator. The
`BatchResult` contract returns Hartree energies and Hartree/angstrom forces
without mutating ASE's single-structure result cache. MACE uses disconnected
graphs, AIMNet2 masks neighbors by molecule index, and ANI batches only
identical element orderings. A fixed 1,270-state parity protocol bounds the
largest batch-versus-serial relative-energy difference at
`2.86e-4 kcal/mol` and the propagated discrete-correction difference at
`3.48e-5 kcal/mol`. These are numerical-equivalence results, not an accuracy
change in \(W_i\). The same evidence records backend-specific speedups of
`0.99x`, `1.42x`, and `12.39x` for MACE-OFF23m, AIMNet2, and ANI2x. MACE
misses the every-paired-repeat `1.25x` floor; AIMNet2 and ANI2x are admitted
for named material-speedup claims, and no universal claim is allowed. The
original relative-energy, propagated-correction, and speed gates
were prospective; protocol v2 added absolute-energy admission at the same
`0.001 kcal/mol` tolerance converted to Hartree and repaired provenance,
warmup, and decision logic after review of the initial run. The final rerun is
therefore review-amended confirmation rather than wholly pre-data evidence.
A preserved v2 run subsequently exposed an unstable MACE speed decision:
its ratio-of-medians exceeded `1.25x`, while paired repeats were `1.060x` and
`1.619x`. Protocol v3 consequently uses four balanced repeats and admits a
per-model material-speed claim only when every paired repeat clears the
unchanged floor. A concurrent-CUDA refresh that produced a 174 s MACE batch
repeat was discarded. Protocol v5 added a fail-closed GPU postflight and a
preflight one-minute host-load ceiling of `0.25` per logical CPU. Review then
noted that preflight/postflight snapshots cannot establish whole-run
exclusivity, so protocol v6 records only endpoint-screening evidence and
explicitly sets continuous monitoring and whole-run exclusivity to false. It
still writes no artifact if either GPU endpoint contains a competing process
or the host preflight exceeds the ceiling.

The frozen state sets are the union of CREST conformers and the original
FreeSolv geometry. Equal weighting of those discrete states assumes equal basin
measure; it does not integrate conformer-basin volumes, vibrational free
energies, a standard-state correction, uncertainty, or an alchemical lambda
path. The result is therefore a fail-closed diagnostic rather than a converged
finite-temperature hydration free energy or a public `#solvfe` result.

## Separately relaxed MLIP transfer diagnostic

The next development-only diagnostic minimizes the gas and implicit-solution
potentials independently:

\[
R_g^\star=\arg\min_R U_{\mathrm{MLIP}}(R),
\qquad
R_s^\star=\arg\min_R[U_{\mathrm{MLIP}}(R)+W(R)].
\]

Its reported 0 K transfer potential is

\[
\Delta E_{\mathrm{relaxed}}
=[U_{\mathrm{MLIP}}(R_s^\star)+W(R_s^\star)]
-U_{\mathrm{MLIP}}(R_g^\star)
=\underbrace{U_{\mathrm{MLIP}}(R_s^\star)
-U_{\mathrm{MLIP}}(R_g^\star)}_{\Delta U_{\mathrm{reorganization}}}
+W(R_s^\star).
\]

MAPLE evaluates both the gas reorganization term and the solvent correction
with the same MACE-OFF23 medium potential, fixed ABCG2 charges, OBC-II polar
term, and ACE nonpolar term. This isolates the intramolecular relaxation that
the MLIP can contribute while retaining an auditable solvent-model boundary.
It is not a free energy: conformer populations, basin entropy, vibrations,
standard-state terms, and lambda sampling are absent.

## Two-level low-force OPT / high-energy final-SP diagnostic

For an energy-only high-accuracy solvent endpoint, Route 1 can use a
force-consistent lower-level solvent potential to generate candidate minima
without claiming that the final-SP model supplied optimization forces. Given
converged low-level gas minima \(R^{\mathrm{low}}_{g,j}\) and solution minima
\(R^{\mathrm{low}}_{s,i}\), the diagnostic reports

\[
\Delta E_{\mathrm{2L}}=
\min_i\left[
U_{\mathrm{MLIP}}(R^{\mathrm{low}}_{s,i})
+W_{\mathrm{high}}(R^{\mathrm{low}}_{s,i})
\right]
-\min_j U_{\mathrm{MLIP}}(R^{\mathrm{low}}_{g,j}).
\]

The MLIP gas potential is retained on both sides; no gas-phase MM energy is
introduced. The candidate geometries are generated with
MACE-OFF23m+AM1-BCC/OBC-II/ACE forces, while
\(W_{\mathrm{high}}\) is fixed-AM1-BCC/CHA-GB/PBSA
cavity-dispersion evaluated only as a single point. Every unique converged
low-level solution minimum is reranked by the displayed high-level total
potential.

This construction is internally well defined only when both potentials and
their distinct derivative capabilities are reported. The current six-case
label-exposed development probe worsens the fixed-geometry high-level MAE from
`1.773` to `1.901 kcal/mol`, with all six cases worsening; it is therefore a
negative benchmark, not a default or a hydration-free-energy claim.

## Energy-only MLIP/solvent Metropolis-TI diagnostic

For an energy-only solvent provider, a conformer-aware free-energy diagnostic
does not require invented solvent forces. Define

\[
U_\lambda(R)=U_{\mathrm{MLIP}}(R)+\lambda W(R),
\qquad
W(R)=G_{\mathrm{polar}}(R,q_{\mathrm{fixed}})
+G_{\mathrm{nonpolar}}(R).
\]

Then

\[
\Delta G_{\mathrm{hyd}}
=A(1)-A(0)
=\int_0^1\left\langle W(R)\right\rangle_\lambda\,d\lambda.
\]

MAPLE's development probe samples each frozen lambda window with symmetric
torsional and Cartesian proposals and the exact Metropolis probability

\[
P_{\mathrm{accept}}
=\min\left[
1,\exp\{-\beta[U_\lambda(R')-U_\lambda(R)]\}
\right].
\]

The MLIP therefore supplies the complete gas conformational potential, while
the fixed-charge provider supplies the solvent endpoint at every proposed
geometry. No MM conformer population, target-force approximation,
experimental residual, or retraining enters. This is a generic
conformer-aware integration pattern, but the current three-case short-chain
probe is a diagnostic rather than a production sampler.

## Conservative fixed-charge GB molecular dynamics

For the admitted connected single-solute NVE/NVT path, MAPLE differentiates
the same fixed-charge potential that it reports:

\[
U_1(R)=U_{\mathrm{MLIP,gas}}(R)+W(R),
\qquad
W(R)=G_{\mathrm{polar}}(R,q_{\mathrm{fixed}})
+G_{\mathrm{nonpolar}}(R).
\]

An NVT trajectory therefore targets
\(\rho_1(R)\propto\exp[-\beta U_1(R)]\), subject to the usual equilibration,
correlation, and finite-sampling limitations. This is a required sampling
foundation, not by itself a solvation-free-energy estimator.

Within the declared implicit model, the configurational free-energy difference
between the solution and gas potentials is

\[
\Delta A_{0\rightarrow1}
=-\beta^{-1}\ln\frac{Z_1}{Z_0}
=-\beta^{-1}\ln\left\langle
\exp[-\beta W(R)]
\right\rangle_{0},
\]

which is the Zwanzig free-energy perturbation identity. Direct endpoint
reweighting is admissible only when the gas ensemble has adequate support for
the solution ensemble. A production implementation must identify equilibrated
and effectively uncorrelated samples, report uncertainty, and fail closed on
insufficient phase-space overlap. Multi-window TI/MBAR is preferred when the
endpoint perturbation lacks overlap. MAPLE does not yet expose this aggregation
as a public `#solvfe` task.

For the production path, define

\[
U_\lambda(R)=U_{\mathrm{MLIP,gas}}(R)+\lambda W(R).
\]

The primary estimate is the MBAR free-energy difference between
\(\lambda=0\) and \(\lambda=1\), while

\[
\Delta G_{\mathrm{TI}}
=\int_0^1 \left\langle W(R)\right\rangle_\lambda\,d\lambda
\]

is an independent cross-check. All sampled configurations must be evaluated
at every lambda value, which is inexpensive for this linear path once
\(U_{\mathrm{MLIP,gas}}(R)\) and \(W(R)\) have been recorded. Endpoint FEP is
diagnostic-only.

For one connected neutral solute, MAPLE defines the model quantity using gas
and ideal-dilute solution standard concentrations of 1 M:

\[
\Delta G_{\mathrm{solv,model}}^{*,1\mathrm{M}\rightarrow1\mathrm{M}}
=-k_BT\ln(Z_1/Z_0).
\]

The same molecular coordinate measure at both endpoints cancels momenta,
center-of-mass translation, the global orientation-group volume, and the
arbitrary additive energy zero of the gas MLIP. Therefore no 1-atm-to-1-M
correction is added to this 1-M-to-1-M result. A comparison or workflow using
a 1-atm gas reference must declare and add that conversion separately.

This exact global-factor cancellation must not be confused with a local
multi-minimum RRHO approximation. When the endpoint configurational integrals
are approximated by selected basins, each basin retains its geometry-dependent
\(Q_{\mathrm{rot},j}/\sigma_{\mathrm{rot},j}\) as part of the local
rovibrational metric. The resulting selected-minimum local-RRHO expression is
a surrogate for the Cartesian endpoint ratio, not a proof that the selected
basins are complete. MAPLE freezes rotational symmetry numbers separately
from conformer degeneracy and never uses duplicate search arrivals as a
statistical weight.

A production analysis must identify an equilibrated suffix, account for
statistical inefficiency, and analyze effectively independent samples. Its
MBAR overlap matrix must connect both endpoints through at least a
tridiagonal chain of adjacent windows. An adjacent off-diagonal element below
`0.03` is treated as a fail-closed caution boundary, not repaired by reporting
only the nominal MBAR uncertainty. Independent repeats, forward/reverse
accumulation, leave-one-window-out sensitivity, and agreement with TI must
also pass prospectively frozen tolerances.

MAPLE delegates this analysis to upstream PyMBAR through the optional
`implicit-free-energy` extra. The public Python API in
`maple.function.free_energy` validates the state/replicate reduced-potential
contract, calls PyMBAR for equilibration detection, decorrelation, MBAR,
effective sample counts, and overlap, and then applies explicit fail-closed
gates. It does not ship a local MBAR implementation. The first nine-record
post hoc exercise validates that analysis plumbing but fails the equilibrium
and per-state independent-sample gates, so it is not a model hydration-free-
energy result.

## Energy-only solvent-provider endpoint correction

An energy-only high-accuracy fixed-charge provider can be applied after
sampling with a force-capable Route 1 provider without changing the gas
Hamiltonian. Define

\[
U_{\mathrm{low}}(R)
=U_{\mathrm{MLIP,gas}}(R)+W_{\mathrm{OBC2/ACE}}(R),
\]

\[
U_{\mathrm{high}}(R)
=U_{\mathrm{MLIP,gas}}(R)+W_{\mathrm{CHA\mbox{-}GB/PBSA}}(R).
\]

For an equilibrated low-endpoint solution ensemble, the exact one-sided
correction is

\[
\Delta F_{\mathrm{low}\rightarrow\mathrm{high}}^{\mathrm{solution}}
=-RT\ln\left\langle
\exp\{-\beta[
W_{\mathrm{CHA\mbox{-}GB/PBSA}}(R)
-W_{\mathrm{OBC2/ACE}}(R)]\}
\right\rangle_{\mathrm{low}}.
\]

The corrected result is

\[
\Delta G_{\mathrm{solv}}^{\mathrm{high}}
=\Delta G_{\mathrm{solv}}^{\mathrm{low}}
+\Delta F_{\mathrm{low}\rightarrow\mathrm{high}}^{\mathrm{solution}}.
\]

There is no gas-leg correction: both endpoints use the identical selected
gas-phase MLIP and the solvent-provider difference is zero in gas. This is not
a residual model or an MM substitution. Although the gas-MLIP energy cancels
from the same-geometry energy difference, it still determines the sampled
low-endpoint ensemble and therefore the perturbation average.

The reusable
`maple.function.free_energy.analyze_one_sided_perturbation` API delegates EXP
and time-series analysis to PyMBAR and reports effective weights, independent
replicates, and fail-closed numerical and scientific gates. The first frozen
three-MLIP/three-case exercise evaluated 720 CHA-GB/PBSA endpoints in
`21.64 s`. Observed weights did not collapse: the minimum combined effective
sample fraction is `0.911` and the maximum normalized weight is `0.136`.
Nevertheless, equilibration detection leaves only `4-13` selected samples per
replicate from the 30-fs parent trajectories, so `0/9` records pass the
predeclared numerical gate and `0/9` pass the scientific gate. The apparent
three-case MAE gains are development diagnostics only and cannot establish
accuracy, target-ensemble coverage, or a public solvation-free-energy result.

The broader 19-case screen applies the same high solvent endpoint directly to
the frozen common discrete states:

\[
\Delta G_{\mathrm{disc}}(W_{\mathrm{high}})
=-RT\ln
\frac{\sum_i\exp\{-\beta[
E_{\mathrm{MLIP,gas}}(R_i)+W_{\mathrm{high}}(R_i)]\}}
{\sum_i\exp\{-\beta E_{\mathrm{MLIP,gas}}(R_i)\}}.
\]

The gas MLIP therefore contributes the conformer weights without being
replaced or fitted. Two exact repeats evaluate 1,270 states for three MLIPs,
or 2,540 CHA-GB/PBSA provider calls. The label-free
[`route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json)
is scored separately in
[`route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json).
Fixed high-endpoint geometry has `1.700 kcal/mol` MAE; discrete high-endpoint
weighting gives `1.699`, `1.720`, and `1.709 kcal/mol` for AIMNet2, ANI2x, and
MACE-OFF23m. Every paired interval versus fixed high-endpoint geometry crosses
zero, only `6-7/19` cases improve, and no model passes every weight diagnostic.
The frozen decision is `long_sampling_candidate_not_supported`: the accuracy
gain over low-endpoint discrete results is attributable to the high fixed-
charge provider, not to a demonstrated MLIP conformer-population advantage.

## MM/GB reference sampling with an unchanged MLIP target

A cheap molecular-mechanics potential may be used as a sampling reference
without entering Route 1's reported target Hamiltonian. Define, separately in
gas and solution,

\[
U_p^{\mathrm{ref}}(R),\qquad
U_p^{\mathrm{target}}(R),
\qquad p\in\{\mathrm{gas},\mathrm{solution}\},
\]

with

\[
U_{\mathrm{gas}}^{\mathrm{target}}(R)
=U_{\mathrm{MLIP,gas}}(R),
\qquad
U_{\mathrm{solution}}^{\mathrm{target}}(R)
=U_{\mathrm{MLIP,gas}}(R)+W(R).
\]

The exact thermodynamic cycle is

\[
\Delta G_{\mathrm{solv}}^{\mathrm{target}}
=\Delta G_{\mathrm{solv}}^{\mathrm{ref}}
+\left(F_{\mathrm{solution}}^{\mathrm{target}}
      -F_{\mathrm{solution}}^{\mathrm{ref}}\right)
-\left(F_{\mathrm{gas}}^{\mathrm{target}}
      -F_{\mathrm{gas}}^{\mathrm{ref}}\right).
\]

For a reference-only endpoint correction,

\[
\Delta F_p^{\mathrm{ref}\rightarrow\mathrm{target}}
=-\beta^{-1}\ln\left\langle
\exp\{-\beta[U_p^{\mathrm{target}}(R)
-U_p^{\mathrm{ref}}(R)]\}
\right\rangle_{\mathrm{ref},p}.
\]

This EXP estimate requires only target energies on reference configurations.
It is nevertheless overlap-limited. Route 1 therefore uses bidirectional
PyMBAR MBAR as the primary validation estimator, PyMBAR BAR as a two-state
cross-check, and EXP only as the prospective accelerated estimate. The
absolute MM-to-MLIP corrections can be hundreds of thousands of kcal/mol
because the models have unrelated additive energy zeros; only the
gas/solution cycle difference is physical.

The three cycle terms reuse sampled configurations. Route 1 therefore does not
combine their individual uncertainty estimates by naive quadrature. It retains
the component uncertainties and requires a joint, chain-aware covariance or
resampling analysis before any cycle-level confidence interval is reported.

The first label-free 3x3 diagnostic confirms exact provider parity but rejects
single-step reference-only reweighting. Minimum gas and solution BAR overlaps
are `0.015` and `0.057`; the corresponding minimum directional MBAR overlap
entries are still lower at `0.005` and `0.016`. Every forward/reverse agreement
check fails, only `7/9` records pass all explicit solver-convergence checks,
and the largest accelerated-versus-bidirectional cycle difference is
`2.514 kcal/mol`. Final PyMBAR nonconvergence messages are stored in the raw
diagnostics and force a failed numerical gate. The method is formally
MLIP-agnostic but practically overlap-limited. A failed endpoint correction
must add intermediate Hamiltonians or use validated nonequilibrium switching
rather than silently returning a free energy.

### Bidirectional nonequilibrium switching

For the force-based reference-to-target bridge, MAPLE uses a linear
Hamiltonian in each phase:

\[
U_{p,\lambda}(R)=(1-\lambda)U_p^{\mathrm{ref}}(R)
+\lambda[U_p^{\mathrm{target}}(R)-C],
\qquad 0\leq\lambda\leq1.
\]

The offset
\(C=U_{\mathrm{target,gas}}(R_{\mathrm{anchor}})
-U_{\mathrm{ref,gas}}(R_{\mathrm{anchor}})\)
is constant in coordinate space and identical in gas and solution. Therefore
it changes neither endpoint force and cancels from
\(\Delta F_{\mathrm{solution}}-\Delta F_{\mathrm{gas}}\). Phase-specific
alignment is forbidden.

Each integration step propagates at \(\lambda_t\), then perturbs the
Hamiltonian at the propagated coordinate:

\[
\delta W_t=
U_{p,\lambda_{t+1}}(R_{t+1})
-U_{p,\lambda_t}(R_{t+1})
=(\lambda_{t+1}-\lambda_t)
[U_p^{\mathrm{target}}(R_{t+1})-C-U_p^{\mathrm{ref}}(R_{t+1})].
\]

The public `maple.function.free_energy` API exposes a model-agnostic endpoint
evaluator protocol, a force-consistent linear calculator, an LFMiddle
switching runner, and a PyMBAR EXP/BAR analysis. Numerical work-distribution
gates are kept separate from endpoint-equilibrium and independent-start
claims.

The prospective 3x3 experiment used only eight work values per direction and
5-fs/20-fs switches. It passes neither a production-length nor endpoint-
equilibrium claim. Its minimum gas/solution BAR overlaps are `0.087/0.072`,
but 5-fs-to-20-fs cycle changes reach `1.364 kcal/mol`, no complete numerical
gate set passes, and the target-force count is `29,376`. Development MAE is
`2.330 kcal/mol` overall, versus `2.150` for fixed geometry, `2.665` for the
failed endpoint correction, and `2.213` for the short direct-target MBAR
control. Thus switching partially repairs the estimator, not the implicit-
solvent model.

References:

- Zwanzig, “High-Temperature Equation of State by a Perturbation Method. I.
  Nonpolar Gases,” *J. Chem. Phys.* 1954, DOI `10.1063/1.1740409`.
- Shirts and Chodera, “Statistically Optimal Analysis of Samples from Multiple
  Equilibrium States,” *J. Chem. Phys.* 2008,
  DOI `10.1063/1.2978177`.
- Klimovich, Shirts, and Mobley, “Guidelines for the Analysis of Free Energy
  Calculations,” *J. Comput.-Aided Mol. Des.* 2015,
  DOI `10.1007/s10822-015-9840-9`.
- Matos et al., “Approaches for Calculating Solvation Free Energies and
  Enthalpies Demonstrated with an Update of the FreeSolv Database,”
  *J. Chem. Eng. Data* 2017, DOI `10.1021/acs.jced.7b00104`.
- Mobley, Dill, and Chodera, “Treating Entropy and Conformational Changes in
  Implicit Solvent Simulations of Small Molecules,” *J. Phys. Chem. B* 2008,
  DOI `10.1021/jp0764384`.
- Chodera, “A Simple Method for Automated Equilibration Detection in Molecular
  Simulations,” *J. Chem. Theory Comput.* 2016,
  DOI `10.1021/acs.jctc.5b00784`.
- König et al., “Multiscale Free Energy Simulations: An Efficient Method for
  Connecting Classical MD Simulations to QM or QM/MM Free Energies Using
  Non-Boltzmann Bennett Reweighting Schemes,” *J. Chem. Theory Comput.* 2014,
  DOI `10.1021/ct401118k`.
- Jia et al., “Calculations of Solvation Free Energy through Energy
  Reweighting from Molecular Mechanics to Quantum Mechanics,”
  *J. Chem. Theory Comput.* 2016, DOI `10.1021/acs.jctc.5b00920`.
- Tkaczyk et al., “Reweighting from Molecular Mechanics Force Fields to the
  ANI-2x Neural Network Potential,” *J. Chem. Theory Comput.* 2024,
  DOI `10.1021/acs.jctc.3c01274`.
- Schöller et al., “Optimizing the Calculation of Free Energy Differences in
  Nonequilibrium Work SQM/MM Switching Simulations,” *J. Phys. Chem. B* 2022,
  DOI `10.1021/acs.jpcb.2c00696`.
- Karwounopoulos et al., “Insights and Challenges in Correcting Force Field
  Based Solvation Free Energies Using a Neural Network Potential,”
  *J. Phys. Chem. B* 2024, DOI `10.1021/acs.jpcb.4c01417`.
- Wang et al., “Accurate Free Energy Calculation via Multiscale Simulations
  Driven by Hybrid Machine Learning and Molecular Mechanics Potentials,”
  *J. Chem. Theory Comput.* 2025, DOI `10.1021/acs.jctc.5c00598`.

## QEq-GTO

This section documents an explicit research control. Fixed QEq-GTO may be used
as a nondefault fixed-charge diagnostic; polarizable CQEq-GTO/GB is outside the
Route 1 fixed-charge product contract.

MAPLE's `qeq-gto` profile uses the original QEq atomic parameters and
hydrogen self-consistency, with a specified single-Gaussian approximation to
the original Slater charge densities.  For each atom,

\[
\zeta_i^0=\lambda\frac{2n_i+1}{2R_i^{\mathrm{bohr}}},
\qquad \lambda=0.5,
\]

where (R_i) is the UFF covalent radius and (n_i) is the principal
quantum number.  The fitted Gaussian exponent and two-center integral are

\[
\alpha_i=c_{n_i}\frac{\zeta_i^2}{n_i},\qquad
\beta_{ij}=\sqrt{\frac{2\alpha_i\alpha_j}{\alpha_i+\alpha_j}},
\]

\[
J_{ij}^{\mathrm{GTO}}(r)=E_h
\frac{\operatorname{erf}(\beta_{ij}r^{\mathrm{bohr}})}{r^{\mathrm{bohr}}}.
\]

The fitting coefficients are (c_1=0.270917), (c_2=0.098800),
(c_3=0.055600), (c_4=0.039100), and (c_5=0.029600).  The runtime
domain is H, C, N, O, F, P, S, Cl, Br, and I.

Hydrogen is not frozen at its neutral-atom parameters.  Following equations
20--21 of Rappé and Goddard,

\[
\zeta_H(q_H)=\zeta_H^0+q_H,
\qquad
J_{HH}(q_H)=J_{HH}^0\left(1+\frac{q_H}{\zeta_H^0}\right).
\]

The corresponding published electrostatic energy is

\[
E_{\mathrm{QEq}}=
\sum_{i\notin H}\left(\chi_i^0q_i+\frac12J_{ii}^0q_i^2\right)
+\sum_{i\in H}\left[\chi_i^0q_i+\frac12J_{ii}^0
\left(1+\frac{q_i}{\zeta_H^0}\right)q_i^2\right]
+\sum_{i<j}q_iq_jJ_{ij},
\qquad \sum_iq_i=Q.
\]

Every SCF iteration rebuilds the hydrogen diagonal and every H--X/H--H pair
integral, solves the charge-constrained KKT system, then mixes the new charges
with damping 0.4.  MAPLE fails closed outside the published
(-1<q_H<1) range, on non-convergence, or when the charge/KKT residual gates
fail.  This repairs both simplifications explicitly documented by Open Babel;
Caltech `cheq` 0.5.1 supplies the permissively licensed GTO mapping and
parameter cross-check, but its fixed pair matrix is not used as the complete
hydrogen-SCF algorithm.

This is a formula-locked **QEq-GTO profile**, not the exact STO numerical
model reported in the 1991 tables.  MAPLE does not relabel its GTO regression
charges as the original paper's STO charge benchmarks.

The original QEq SCF neglects the full charge derivative of the
charge-dependent hydrogen exponent.  It is therefore not a strict stationary
point of the displayed QEq energy and is used only for `mode=fixed`.

For `mode=polarizable`, MAPLE implements the consistent-QEq derivative from
Ogawa et al.  For a hydrogen (i), this adds the derivative of the cubic
one-center term and every hydrogen-dependent pair integral:

\[
\frac{\partial E}{\partial q_i}=\chi_i^0+J_{ii}^0q_i
+\frac{3J_{ii}^0}{2\zeta_H^0}q_i^2
+\sum_{j\ne i}\left(J_{ij}+q_i\frac{\partial J_{ij}}{\partial q_i}\right)q_j.
\]

The GB polar contribution adds (K_{GB}q).  MAPLE solves the resulting
nonlinear constrained minimum with an analytic charge gradient, then requires
charge conservation, a KKT residual below `2e-6 eV`, and a nonnegative
projected charge-Hessian eigenvalue.  Only this CQEq path uses the envelope
theorem for polarizable forces.  This is a mathematically consistent but still
experimental continuum coupling; its parameters were not jointly trained with
the Amber GB radii or nonpolar term.

References:

- A. K. Rappé and W. A. Goddard III, “Charge equilibration for molecular
  dynamics simulations,” *J. Phys. Chem.* **95** (1991),
  DOI `10.1021/j100161a070`.
- J. Chen and T. J. Martínez, “Charge conservation in electronegativity
  equalization and its implications for the electrostatic properties of
  fluctuating-charge models,” *J. Chem. Phys.* **131**, 044114 (2009),
  DOI `10.1063/1.3183167`.
- T. Ogawa et al., “Consistent Charge Equilibration Method Combined with
  Universal Force Field: Application to Amino Acid Molecules,”
  *Chem-Bio Informatics Journal* **3**, 78--85 (2003),
  DOI `10.1273/cbij.3.78`, for the explicit consistency problem and full
  charge derivatives required by a variational QEq force model.
- L. G. Dias, K. Shimizu, J. P. S. Farah, and H. Chaimovich, “A simple method
  for the fast calculation of charge redistribution of solutes in an implicit
  solvent model,” *Chemical Physics* **282** (2002), DOI
  `10.1016/S0301-0104(02)00717-6`, for the self-consistent QEq/continuum
  precedent.
- Caltech `cheq` 0.5.1, MIT license, commit
  `f71eeb4ab3790557fb244a8f09fa60e255ee6274`, for the UFF-radius parameter
  table and the explicitly versioned STO-to-GTO fitting coefficients.
- Open Babel `qeq.cpp` at commit
  `0e94434fa75c9f61095023e3c12e0d5f2ac035ff`, reviewed as a rejected
  simplification because its own documentation states that charge-dependent
  hydrogen hardness and screening are omitted.

## Amber GB models

MAPLE calls OpenMM's upstream `CustomAmberGBForceBase` implementations and
their `getStandardParameters(topology)` radius assignment.  It does not copy the
HCT/OBC/GBn descreening equations.

The primary hydration benchmark uses OpenMM's ACE term, exactly as implemented
by `customgbforces._createEnergyTerms`:

\[
G_{np}^{\mathrm{ACE}}=\sum_i 28.3919551\,(r_i+0.14)^2
\left(\frac{r_i}{B_i}\right)^6,
\]

with OpenMM's internal nanometre/kJ-mol unit convention.  This term is traced to
Schaefer and Karplus, “A Comprehensive Analytical Treatment of Continuum
Electrostatics,” *J. Phys. Chem.* (1996), DOI `10.1021/jp9521621`.

ACE is **not** relabelled as Amber `gbsa=1`.  Amber documents `gbsa=1` as LCPO
with

\[
G_{np}^{\mathrm{LCPO}}=\gamma A_{\mathrm{LCPO}},\qquad
\gamma=0.005\ \mathrm{kcal\,mol^{-1}\,\mathring A^{-2}}.
\]

Therefore independent complete-energy parity uses OpenMM 8.5's separate
`LCPOForce`, while experimental-accuracy ranking continues to use the
predeclared ACE profile.  Mixing those two nonpolar definitions would be a
model change, not a numerical tolerance.

The fixed-parameter development comparison now quantifies that model change
on every molecule for which OpenMM supplies LCPO parameters. On the identical
454-case subset, OBC-II/ACE has MAE `1.801 kcal/mol` and OBC-II/LCPO has
`2.250 kcal/mol`; the paired ACE-minus-LCPO MAE gain is
`-0.449 kcal/mol` with 95% bootstrap interval `[-0.581, -0.318]`.
The OBC-II polar arrays agree exactly, so this is an isolated nonpolar-model
result rather than a charge or polar-provider change.

For small-molecule parity, the submitted GAFF atom types disambiguate two
upstream topology heuristics. Generic `MOL` ester carbonyl oxygen retains
Amber's `1.50 A` `mbondi3` radius rather than OpenMM's carboxylate-like
connectivity result. LCPO types `o` and `o2` select OpenMM's `O_sp2_1` and
`O_carboxylate` rows respectively, so nitro oxygen is not silently treated as
carboxylate oxygen. These are audited provider translations of the declared
Amber atom types, not hydration-label fits.

OpenMM 8.5.2's generic `GBSAGBn2Force` table has no Amber-specific phosphorus
alpha/beta/gamma entry.  MAPLE consequently rejects GBn2 for P-containing
molecules instead of silently accepting OpenMM's generic default. The generic
OpenMM expression also zeros a close-pair branch that Amber's signed
Taylor-series descreening retains for sulfur's negative screening radius.
MAPLE therefore rejects GBn2 for S-containing molecules until an authoritative
energy-and-force implementation is available. Other GB models still run, and
the benchmark records those method-specific applicability failures in its
original denominator.

- HCT: Hawkins, Cramer, Truhlar, “Pairwise solute descreening of solute charges
  from a dielectric medium,” *Chem. Phys. Lett.* (1995),
  DOI `10.1016/0009-2614(95)01082-K`.
- OBC-I/OBC-II: Onufriev, Bashford, Case, “Exploring protein native states and
  large-scale conformational changes with a modified generalized Born model,”
  *Proteins* (2004),
  DOI `10.1002/prot.20033`.
- GBn: Mongan et al., “Generalized Born model with a simple, robust molecular
  volume correction,” *J. Chem. Theory Comput.* (2007),
  DOI `10.1021/ct600085e`.
- GBn2: Nguyen et al., “Improved Generalized Born Solvent Model Parameters for
  Protein Simulations,” *J. Chem. Theory Comput.* (2013),
  DOI `10.1021/ct3010485`.
- LCPO: Weiser, Shenkin, Still, “Approximate atomic surfaces from linear
  combinations of pairwise overlaps (LCPO),” *J. Comput. Chem.* (1999),
  DOI `10.1002/(SICI)1096-987X(19990130)20:2<217::AID-JCC4>3.0.CO;2-A`.

## Development CHA-GB and cavity/dispersion endpoints

The label-blind development comparison in
`chagb_nonpolar_protocol.json` keeps the frozen AM1-BCC charges and
single FreeSolv geometry, but replaces the OBC-II polar endpoint with
AmberTools 26 GBNSR6 charge-hydration-asymmetric GB:

\[
G_{\mathrm{CHA+surface}}
=E_{\mathrm{GB}}^{\mathrm{CHA}}
+E_{\mathrm{SURF}}.
\]

The frozen CHA controls are `epsin=1`, `epsout=78.5`, zero salt,
`space=0.3 A`, `arcres=0.2 A`, `radiopt=0`, `ROH=0.586 A`, and
`tau=1.47`.  `ESURF` uses the provider's
`0.005 kcal mol^-1 A^-2` surface tension.  This endpoint tests the polar
model change without fitting any term to FreeSolv labels.

The second endpoint replaces only that simple nonpolar term:

\[
G_{\mathrm{CHA+cav/disp}}
=E_{\mathrm{GB}}^{\mathrm{CHA}}
+E_{\mathrm{CAVITY}}
+E_{\mathrm{DISPER}}.
\]

PBSA `inp=2` separates repulsive cavity formation from attractive
solute-water dispersion.  The protocol explicitly freezes the published
TIP3P/PME sigma-decomposition settings: `decompopt=2`, `use_rmin=1`,
`sprob=0.557 A`, `vprob=1.300 A`, `rhow_effect=1.129`, `use_sav=1`,
`cavity_surften=0.0378`, and `cavity_offset=-0.5692 kcal/mol`.
The pinned FreeSolv GAFF atom types are retained while the charge column is
replaced by the frozen AM1-BCC vector; GAFF2 supplies the nonbonded parameters
needed for dispersion.

The same locked endpoint is available through the explicit energy-only runtime
profile `provider=ambertools,model=chagb`. Its reported solution potential is

\[
E_{\mathrm{solution}}(R)
=E_{\mathrm{MLIP,gas}}(R)
+E_{\mathrm{GB}}^{\mathrm{CHA}}(R,q_{\mathrm{AM1-BCC}})
+E_{\mathrm{CAVITY}}(R)
+E_{\mathrm{DISPER}}(R).
\]

`parmchk2` and `tleap` construct the provider topology, but MAPLE parses only
the three displayed solvent components. No bonded, nonbonded gas-phase, or
other MM energy is included. The profile supports SP energy only and rejects
all derivative requests.

### Prepared topology and coordinate-only evaluation

The runtime cache changes the execution graph, not the potential.  For one
provider instance, MAPLE prepares once from the finite coordinates supplied at
provider construction (rather than whichever pose happens to be scored first):

\[
(\text{typed MOL2},q_{\mathrm{fixed}})
\xrightarrow{\mathrm{parmchk2},\ \mathrm{tleap}}
(\mathrm{frcmod},\mathrm{prmtop}).
\]

Later conformers or poses retain those immutable files and replace only the
Amber coordinate input,

\[
(\mathrm{prmtop},R)
\xrightarrow{\mathrm{GBNSR6},\ \mathrm{PBSA}}
E_{\mathrm{GB}}^{\mathrm{CHA}}(R)+E_{\mathrm{CAVITY}}(R)+E_{\mathrm{DISPER}}(R).
\]

The cache fingerprint binds the source-MOL2 SHA256, construction-coordinate
SHA256, fixed-charge-vector SHA256, atom-type-vector SHA256, full
polar/nonpolar profile SHA256, and the AmberTools executable-bundle SHA256. It
is provider-instance-local; it is not
a cross-molecule topology cache, a fitted energy correction, or a persistent
external GBNSR6/PBSA process. The GBNSR6 in-process/POSIX serialization remains
in force.

The live parity artifact
`route1-chagb-prepared-provider-parity-2026-07-29.json` compares this path
against the prior fresh-`parmchk2`/`tleap` workflow on one pinned fixed-AM1-BCC
methyl-hexanoate and four coordinate arrays (including one nonrigid probe).
Every displayed solvent
component and their total agree within `1e-9 kcal mol^-1` (observed maximum
`0.0`). This proves only that narrow execution parity; it does not create
forces or establish a general accuracy, ranking, or throughput claim.

Both energy phases consume a label-free source manifest.  Experimental
FreeSolv values are opened only by the separate summary phase after all 526
energy records exist.  No residual correction, experimental refit, or
per-chemistry selector is present.

### Factorial endpoint attribution

The Route 1 component-attribution audit treats the frozen development MAE as
a four-endpoint response, not as an additional physical energy term. Let
\(m_{00}\) be OBC-II/ACE, \(m_{10}\) CHA-GB/ACE, \(m_{01}\)
OBC-II/PBSA cavity-dispersion, and \(m_{11}\) CHA-GB/PBSA
cavity-dispersion. The total observed MAE gain is

\[
\Delta m = m_{00}-m_{11}.
\]

For a symmetric two-factor description, the endpoint Shapley allocations are

\[
\phi_{\mathrm{polar}}
=\frac{1}{2}\left[(m_{00}-m_{10})+(m_{01}-m_{11})\right],
\]

\[
\phi_{\mathrm{nonpolar}}
=\frac{1}{2}\left[(m_{00}-m_{01})+(m_{10}-m_{11})\right].
\]

They satisfy \(\phi_{\mathrm{polar}}+\phi_{\mathrm{nonpolar}}=\Delta m\).
The reported marginal-gain interaction is

\[
I=(m_{10}-m_{00})-(m_{11}-m_{01}).
\]

For the 526 frozen development cases,
\(\Delta m=0.4385\), \(\phi_{\mathrm{polar}}=0.3611\),
\(\phi_{\mathrm{nonpolar}}=0.0774\), and \(I=0.8404\) kcal/mol. Because MAE is
nonlinear and the radii/nonpolar parameters are coupled, this accounting does
not prove microscopic causality or decompose the physical solvation energy.
It only establishes that neither isolated swap passes while the joint endpoint
does. No Shapley value enters the Route 1 potential, provider selection, or
runtime.

### Pre-registered paired reserve statistic

For the label-exposed 116-case reserve, the endpoint energies were sealed
before labels were read. With signed endpoint errors
\(e_i^{(0)}\) for OBC-II/ACE and \(e_i^{(1)}\) for the coupled
CHA-GB/PBSA endpoint, the pre-registered paired improvement variable is

\[
g_i = \left|e_i^{(0)}\right|-\left|e_i^{(1)}\right|,
\qquad
\overline{g}=\frac{1}{N}\sum_i g_i.
\]

The observed \(\overline{g}\) is `0.4901 kcal/mol`; a 10,000-resample paired
bootstrap gives the 95% interval `[0.2146, 0.7821]`. The frozen retention rule
also requires full coverage, \(\overline{g}\ge 0.15\) kcal/mol, and no
worsening of aggregate RMSE or maximum absolute error. All overall gates pass.
No term involving \(g_i\) is added to the physical potential: it is only a
model-evaluation statistic. Because both endpoint predictions are
fixed-geometry solvent corrections, this benchmark assesses the continuum
correction and charge/radius/nonpolar pairing; the gas-phase MLIP cancels from
the paired hydration score. MLIP quality contributes instead through combined
forces, relaxed geometries, conformer populations, and reorganization terms,
none of which this reserve establishes.

### Series-level relative-ranking diagnostic (not a certification)

Absolute hydration error does not determine whether a same-series direction is
right.  For a predeclared series (s) and members (i,j), Route 1 therefore
measures the uncalibrated score difference

\[
\Delta\Delta G^{\mathrm{pred}}_{s,ij}
=\widehat G_{s,i}-\widehat G_{s,j}
\]

against the experimental difference

\[
\Delta\Delta G^{\mathrm{exp}}_{s,ij}
=G_{s,i}-G_{s,j}.
\]

It reports Kendall \(\tau_b\), Spearman \(\rho\), forced pair-direction
accuracy, and the MAE/RMSE of
\(\Delta\Delta G^{\mathrm{pred}}-\Delta\Delta G^{\mathrm{exp}}\).  A pair is
also reported separately only when it is experimentally distinguishable:

\[
\left|\Delta\Delta G^{\mathrm{exp}}_{s,ij}\right|
>z\sqrt{\sigma_i^2+\sigma_j^2},\qquad z=1.96.
\]

Fixed \(0.5\), \(1.0\), and \(2.0\) kcal/mol difference gates are displayed
alongside that uncertainty gate.  These are free-energy differences, not
percentage relative errors, so they remain well defined near zero and when a
free energy changes sign.  Per-series metrics are macro-averaged,

\[
\overline{\tau}_{\mathrm{macro}}
=\frac{1}{S}\sum_{s=1}^{S}\tau_s,
\]

and bootstrap resampling draws complete series rather than overlapping pairs.
Top-\(k\) overlap/enrichment and best-of-top-\(k\) regret are also reported
only for an unambiguous top-\(k\) boundary.

The first implementation consumes the already frozen 116-record score only
after its label-free energy phase.  It derives seven evaluable, **structure
only** Bemis--Murcko scaffold groups (48 members) from hash-pinned MOL2 files;
the old `structure_group_sha256` field is merely the hash of one molecule's
canonical SMILES and cannot define a series because it is unique for all 642
prepared molecules.  Thus these groups are a water-hydration structural
diagnostic, not a matched-pair proof, a congeneric medicinal-chemistry series,
or a protein--ligand benchmark.

On this explicitly limited diagnostic, AM1-BCC/OBC-II/ACE gives macro
\(\tau_b/\rho=0.093/0.122\), while fixed-geometry
AM1-BCC/CHA-GB/PBSA gives \(0.375/0.433\).  The candidate-minus-baseline
complete-series bootstrap intervals are \([-0.101,0.902]\) for \(\tau_b\) and
\([-0.060,0.911]\) for \(\rho\), so neither interval rules out no improvement.
The result is evidence that direct ranking measurement is necessary; it is
not evidence to promote an endpoint or claim a generally correct trend.

For a future selective ranker, a predeclared margin
\(m_{s,ij}=|\Delta\Delta G^{\mathrm{pred}}_{s,ij}|\) can yield a retrospective
coverage--risk curve, but Route 1 currently emits no `certified` ordering:
there is no series-disjoint calibration set, no external independent test
series, and no fixed conformal threshold.  A future abstention rule must be
calibrated before a series-disjoint test, retain the raw energies unchanged,
and return `uncertain` outside its stated applicability domain.

### Matched-pair charge continuity is a diagnostic, not an energy correction

For one structure-only matched-core mapping \(m\) between molecules \(i\) and
\(j\), let \(\mathcal R_m\) contain only mapped **heavy atoms** that are at
least three heavy-atom bonds from the transformation attachment in both
molecules. The charge-continuity audit measures

\[
D_{q,2}^{(m)}
=
\left[
\sum_{(a,b)\in\mathcal R_m}
\left(q_{i,a}-q_{j,b}\right)^2
\right]^{1/2},
\qquad
D_{q,\mathrm{RMS}}^{(m)}
=\frac{D_{q,2}^{(m)}}{\sqrt{|\mathcal R_m|}},
\]

together with
\(\max_{(a,b)\in\mathcal R_m}|q_{i,a}-q_{j,b}|\). Hydrogens are not included,
so these quantities must not be described as total common-core charge drift.
RDKit `2024.09.2` first returns one `FindMCS` SMARTS from element, bond-order,
and ring topology before any charge or experimental value is examined. The
audit enumerates every non-uniquified embedding of that **single returned
SMARTS** on both molecules; it does not enumerate distinct, non-isomorphic
maximum-MCS pattern alternatives. If those audited embeddings give a nonzero
interval for a reported charge metric, that pair is mapping-ambiguous and is
excluded from the main diagnostic rather than choosing the embedding that best
explains an error.

The reason this norm is useful but cannot become a kcal/mol threshold follows
from the fixed-geometry GB quadratic form. With

\[
G_{\mathrm{polar}}(q)
=-\frac{c}{2}q^\mathsf{T}Kq,
\qquad c=1-\varepsilon_{\mathrm{out}}^{-1},
\]

a charge perturbation \(\delta q\) at a *fixed* geometry and kernel gives

\[
G_{\mathrm{polar}}(q+\delta q)-G_{\mathrm{polar}}(q)
=-c\,q^\mathsf{T}K\delta q
-\frac{c}{2}\delta q^\mathsf{T}K\delta q,
\]

and therefore

\[
\left|\delta G_{\mathrm{polar}}\right|
\le
c\,\lVert Kq\rVert_2\lVert\delta q\rVert_2
+\frac{c}{2}\lVert K\rVert_2\lVert\delta q\rVert_2^2.
\]

A real matched pair changes atom count, geometry, radii, and thus the kernel
\(K\); the audit also observes only a remote heavy-atom subvector. Without a
uniform kernel bound and an independently calibrated series distribution,
\(D_q\) cannot be converted into an energy-error bar, an abstention threshold,
or a causal decomposition. It is only a representation-fragility feature to
test against held-out series.

The frozen 116-case reserve contains complete AM1-BCC vectors for both Route 1
endpoints but no same-record ABCG2, RESP, or RESP2 vectors. Consequently this
audit can test **within-AM1-BCC continuity** and OBC-II-versus-CHA endpoint
order disagreement only. It cannot be reported as a charge-model comparison,
and any later alternate-charge study must be generated and frozen as a new
label-free artifact before opening its evaluation labels.

The implemented audit keeps that boundary physical. A first process reads only
the frozen source manifest, hash-pinned MOL2 files, AM1-BCC vectors, and sealed
endpoint energies. For the single SMARTS returned by RDKit `FindMCS`, it
enumerates all non-uniquified embedding pairs subject to an exact
element/bond-order, complete-ring, connected-core, single-attachment, `1..3`
changed-heavy-atom, and remote-distance contract. The embedding cross-product
is capped; timeout or overflow fails closed. All structure-valid embeddings of
that returned SMARTS must have identical remote-heavy-atom count and charge
metrics before the lexicographically first embedding may represent the pair.
Distinct maximum-MCS SMARTS patterns are an unaudited limitation, not silently
treated as invariant. Mapping-variant embeddings remain in a label-free
sensitivity appendix and are excluded from scoring. Only a second process may
open the already frozen score artifact, after checking the complete
source--energy--score and pair-protocol--pair-artifact hash chains.

Of `6670` possible molecule pairs, `65` satisfy this strict label-free
contract and form five dependent graph components with `(members, edges)`
equal to `(13,52)`, `(6,6)`, `(4,5)`, `(2,1)`, and `(2,1)`. Nine candidate
pairs fail the mapping-enumeration cap; one has a charge-metric-variant
mapping and one has a remote-count-variant mapping. One of the 65 accepted
pairs is an exact experimental tie, leaving 64 experimentally distinct edges.
The retrospective label-open aggregates are:

| fixed experimental gate | edges | OBC-II/ACE sign accuracy | CHA-GB/PBSA sign accuracy | OBC-II/ACE \(\Delta\Delta G\) MAE | CHA-GB/PBSA \(\Delta\Delta G\) MAE |
|---|---:|---:|---:|---:|---:|
| nonzero \(|\Delta\Delta G_{\rm exp}|\) | 64 | 0.953 | 0.922 | 1.088 | 1.187 |
| \(\ge 0.5\) kcal/mol | 59 | 0.949 | 0.949 | 1.151 | 1.230 |
| \(\ge 1.0\) kcal/mol | 47 | 1.000 | 1.000 | 1.276 | 1.337 |
| \(\ge 2.0\) kcal/mol | 39 | 1.000 | 1.000 | 1.475 | 1.547 |
| \(>1.96\sqrt{\sigma_i^2+\sigma_j^2}\) | 41 | 1.000 | 1.000 | 1.266 | 1.438 |

All errors are kcal/mol. The endpoints disagree in direction on six of the 64
distinct edges, four of the `0.5`-kcal/mol edges, and none of the `1.0`,
`2.0`, or uncertainty-gated edges. These observations are not independent:
most edges share molecules inside the 13-member component. No p-value,
confidence interval, regression, bootstrap, or endpoint comparison decision is
therefore reported.

Most importantly for the bottleneck hypothesis, the descriptive Spearman
association between \(D_{q,2}\) and absolute \(\Delta\Delta G\) error is only
`0.158` for OBC-II/ACE and `-0.025` for CHA-GB/PBSA across the 64 distinct
edges. At the predeclared uncertainty gate it is `0.062` and `0.007`.
The RMS and maximum-absolute charge-drift variants lead to the same qualitative
boundary. This frozen subset therefore does **not** establish remote
within-AM1-BCC heavy-atom drift as the dominant CHA ranking bottleneck and
does not support a \(D_q\)-based abstention threshold. The exact conclusion is
`diagnostic_signal_only_insufficient_for_threshold_or_certification`.

### Published explicit-component comparison is a bottleneck diagnostic, not a target

The frozen water endpoint can also be compared with the *published calculated*
FreeSolv decomposition on the same 526 development identifiers.  This is not
a comparison to a second experimental label: `database.json` reports the
FreeSolv GAFF/AM1-BCC explicit-solvent calculation and its charging and van der
Waals legs.  Let those published values be
\(G_{\mathrm{calc}}\), \(G_{\mathrm{chg}}\), and
\(G_{\mathrm{vdW}}\), and let the fixed Route 1 components be
\(G_{\mathrm{CHA}}\) and \(G_{\mathrm{PBSA,np}}\).  The source rounds

\[
G_{\mathrm{calc}}
=G_{\mathrm{chg}}+G_{\mathrm{vdW}}+r_{\mathrm{round}},
\qquad |r_{\mathrm{round}}|\leq 0.0011\ \mathrm{kcal\ mol^{-1}}.
\]

The diagnostic reports the *descriptive* component differences

\[
\delta_{\mathrm{polar}}
=G_{\mathrm{CHA}}-G_{\mathrm{chg}},
\qquad
\delta_{\mathrm{np}}
=G_{\mathrm{PBSA,np}}-G_{\mathrm{vdW}},
\]

and their total, with the rounding convention explicit,

\[
\widehat G_{\mathrm{Route\ 1}}-G_{\mathrm{calc}}
=\delta_{\mathrm{polar}}+\delta_{\mathrm{np}}-r_{\mathrm{round}}.
\]

For experimental hydration value \(G_{\mathrm{exp}}\), the exact accounting
identity is

\[
\underbrace{\widehat G_{\mathrm{Route\ 1}}-G_{\mathrm{exp}}}_{e_{\mathrm{Route\ 1}}}
=
\underbrace{G_{\mathrm{calc}}-G_{\mathrm{exp}}}_{e_{\mathrm{published\ explicit}}}
+
\underbrace{\left(\widehat G_{\mathrm{Route\ 1}}-G_{\mathrm{calc}}\right)}_{\delta_{\mathrm{continuum-vs-explicit}}}.
\]

It is an identity, **not** a causal error decomposition: the published
calculation has its own charge, force-field, conformational, finite-sampling,
and standard-state approximations.  Consequently neither \(\delta\) is a
residual to add to Route 1, nor is \(G_{\mathrm{calc}}\) a teacher or a
replacement target.

The hash-pinned replay
`benchmarks/route1-freesolv-explicit-component-diagnostic-2026-07-29.json`
checks that identity to `1.8e-15 kcal mol^-1`.  Its CHA polar mismatch has
MAE/RMSE/bias/max `0.879/1.296/-0.556/7.629 kcal mol^-1`; the nonpolar mismatch
has `0.426/0.586/+0.083/2.752`.  The full continuum-versus-published-explicit
difference has MAE/RMSE/max `0.936/1.280/7.881`.  On the identical charge and
geometry inputs, OBC-II polar versus published charging is markedly worse
(`2.070/2.743/-1.891/12.285`), while CHA reduces paired absolute polar mismatch
by `1.190 kcal mol^-1` on average (458/526 cases; 147/170 Route 1 tails).
Therefore the evidence retains CHA rather than just reverting a radius model,
but it does **not** prove that the residual CHA discrepancy is causal or that
any particular nonlinear replacement is valid.  A scalar nonpolar change is
likewise not supported: polar discrepancy is larger in 357/526 cases and in
125/170 cases where Route 1 itself exceeds the predeclared `1.5 kcal mol^-1`
absolute-error gate.  The descriptive route-error associations are
Pearson/Spearman `0.551/0.548` for polar mismatch but only `0.039/0.077` for
nonpolar mismatch.  These associations carry no p-values, confidence
intervals, regression, or causal interpretation.

The same identity prevents an overclaim in the opposite direction.  The
published explicit calculation itself exceeds `1.5 kcal mol^-1` on 123/526
records (MAE/RMSE/max `1.135/1.568/10.775`), whereas Route 1 exceeds it on
170/526.  The fixed-gate table contains 71 records where both exceed, 99 Route
1-only, 52 published-explicit-only, and 304 neither.  Therefore merely
approaching this particular explicit calculation cannot by itself satisfy the
user's per-record experimental gate; conversely, the 52 published-explicit-only
records prove that Route 1 is not simply a degraded copy of it.  Any next
physical hypothesis must be pre-registered and independently evaluated against
experiment without endpoint selection, charge switching, parameter fitting, or
an energy correction.

### Scale-controlled surface-electrostatics falsification

The component diagnostic above motivated, but did not establish, a missing
nonlinear-interface or first-shell mechanism.  A separate label-free process
therefore evaluates only four quantities on a deterministic union-sphere SAS
quadrature.  For accessible surface points \(s_k\), areas \(a_k\), outward
normals \(n_k\), solute charges \(q_i\), and atomic positions \(R_i\),

\[
A=\sum_k a_k,
\qquad
\phi_k=\sum_i\frac{q_i}{\lVert s_k-R_i\rVert},
\]

\[
E_{n,k}
=
\sum_i q_i
\frac{(s_k-R_i)\mathbin{\cdot}n_k}
{\lVert s_k-R_i\rVert^3},
\]

\[
\Phi_2
=
\left(\frac{1}{A}\sum_k a_k\phi_k^2\right)^{1/2},
\qquad
F_{n,2}
=
\left(\frac{1}{A}\sum_k a_kE_{n,k}^2\right)^{1/2}.
\]

With
\(\overline E_n=A^{-1}\sum_k a_kE_{n,k}\), the signed heterogeneity proxy is

\[
\Gamma_n
=
\frac{
A^{-1}\sum_k a_k(E_{n,k}-\overline E_n)^3
}{
\left[
A^{-1}\sum_k a_k(E_{n,k}-\overline E_n)^2
\right]^{3/2}
}.
\]

These are geometry/electrostatics descriptors, not a solvent free-energy
functional.  In particular, the SAS proxy is neither the actual CHA/GBNSR6
dielectric interface nor an explicit first solvent shell.

The raw development-set statistic uses

\[
Y_i
=
\left|
G_{\mathrm{CHA,p},i}-G_{\mathrm{published,charging},i}
\right|,
\qquad
Z_i=\left|G_{\mathrm{CHA,p},i}\right|.
\]

Because \(F_{n,2}\), \(\Phi_2\), and \(Z\) all measure electrostatic scale, a
raw rank association does not identify an extra nonlinear mechanism.  The
post-v1 adversarial statistic rank-transforms every variable and residualizes
both \(Y\) and a candidate proxy \(F\) against

\[
X=[\mathbf 1,\operatorname{rank}(Z),\operatorname{rank}(A)].
\]

Writing \(P_X=XX^+\), where \(X^+\) is the Moore--Penrose pseudoinverse, its
point statistic is

\[
\rho_{\mathrm{partial}}
=
\operatorname{corr}
\left[
(I-P_X)\operatorname{rank}(Y),
(I-P_X)\operatorname{rank}(F)
\right].
\]

Cluster-bootstrap replicates use the exactly equivalent count-weighted
least-squares projection after resampling complete
`structure_group_sha256` units.  In this artifact all 526 realized groups are
singletons, so it remains a molecule-level development sensitivity analysis,
not a series-disjoint confirmation.  Bonferroni intervals cover the two
predeclared proxies within each radius view.

The unadjusted Bondi-family correlations are `0.512` for \(F_{n,2}\) and
`0.488` for \(\Phi_2\), while
\(\rho(F_{n,2},Z)=0.822\),
\(\rho(\Phi_2,Z)=0.779\), and
\(\rho(F_{n,2},\Phi_2)=0.960\).  After controlling \(Z\) and \(A\), the
Bondi-family partial values collapse to `0.061` and `0.062`, with simultaneous
95% intervals `[-0.053,0.173]` and `[-0.044,0.170]`; mbondi2 gives `0.059`
`[-0.055,0.169]` and `0.062` `[-0.045,0.169]`.  All intervals cross zero.
The \(\Gamma_n\) interval also crosses zero, and 464/526 radius vectors are
identical between the two radius views, so their agreement is not independent
replication.

The mathematically supported decision is therefore narrow: **stop the
inference from these surface proxies to a new first-shell/nonlinear provider**.
This does not prove that every nonlinear interface theory is false.  It means
that this post-result, label-exposed development sensitivity cannot identify
one, select a radius or endpoint, modify an energy, certify a rank, or support
the requested maximum-error/multi-solvent claims.

### Fail-closed split-conformal selective ranking

`benchmarks/selective_ranking.py` implements the decision contract but does
not activate it for the current FreeSolv scaffold diagnostic. A qualifying
calibration artifact has one complete-series score

\[
R_s=\max_{i<j}
\left|
\Delta\Delta G^{\mathrm{pred}}_{s,ij}
-\Delta\Delta G^{\mathrm{exp}}_{s,ij}
\right|,
\]

never an overlapping-pair sample. For \(m\) exchangeable calibration series
within one exact declared domain and desired miscoverage \(\alpha\), the
implemented split-conformal radius is the order statistic

\[
q_{1-\alpha}
=R_{(\lceil(m+1)(1-\alpha)\rceil)},
\qquad
\lceil(m+1)(1-\alpha)\rceil\le m.
\]

If the displayed finite-sample condition fails, no finite empirical radius can
support the requested target coverage: the implementation returns
`out_of_domain` instead of clamping the rank to \(m\).  In particular,
\(\alpha<1/(m+1)\) cannot be advertised from only \(m\) calibration series.

For an already computed pair difference
\(d=\widehat G_{\mathrm{first}}-\widehat G_{\mathrm{second}}\), it returns
the *interval* \([d-q,d+q]\), without changing \(d\). It certifies
`first_better` only if \(d<-q\), `second_better` only if \(d>q\), and returns
`uncertain` when the interval crosses zero. An absent qualified calibration or
an unseen/undersupported exact domain returns `out_of_domain`. A charge-scheme
or endpoint-order disagreement also forces `uncertain`; it never averages the
disagreeing scores.

The coverage statement is conditional on exchangeability of complete series in
that declared domain. The calibration artifact binds \(\alpha\), the minimum
domain sample size, endpoint fingerprint, domain schema, calibration/test
manifests, explicit held-out series IDs, and a series-disjointness audit before
the calibrator is constructed.  Each prediction must then arrive as a sealed,
label-blind evidence artifact whose series ID belongs to that held-out
manifest, and whose test-energy, endpoint, domain-schema, and disagreement
protocol fingerprints exactly match the calibration contract.  Its full
content hash must also have been pre-enumerated in the calibration-bound
prediction-evidence manifest, so resealing a different delta or ligand pair
does not make it trusted.  A calibration series, an unregistered series, or a
delta from another endpoint fails closed.
Those hashes make silent prediction-time retuning and scalar substitution
detectable; they do not independently prove that an external data source or
the hashed disagreement computation is scientifically honest.
The current FreeSolv structure groups satisfy none of the activation
conditions. Thus the code can refuse safely today, but no `certified` Route 1
prediction is currently scientifically authorized.

### `route1-physdistill` research-only pre-registration

`benchmarks/route1_physdistill_protocol_v1.json` records a separate proposed
research branch; it is intentionally not imported by any Route 1 runtime. The
candidate keeps an analytic GB/CHA long-range backbone and permits learning
only bounded physical intermediates, for example

\[
\widetilde R_i
=R_i^{\mathrm{intrinsic}}
+\left(R_i^{\max}-R_i^{\mathrm{intrinsic}}\right)
\sigma\!\left(f_{\theta,i}\right),
\qquad R_i^{\max}>R_i^{\mathrm{intrinsic}}>0,
\]

not an arbitrary total-energy residual. Its potential must remain a scalar
\(G_\theta(R,q_{\mathrm{fixed}})\) with
\(F_\theta=-\nabla_R G_\theta\). A future teacher-distillation loss may combine
energy, conservative force, and fixed pair-difference terms,

\[
\mathcal L=
\lambda_E\lVert G_\theta-G_T\rVert^2
+\lambda_F\lVert\nabla_R G_\theta-\nabla_R G_T\rVert^2
+\lambda_\Delta\sum_{a,b}
\left|\left(G_\theta^a-G_\theta^b\right)
-\left(G_T^a-G_T^b\right)\right|^2
+\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{limits}}.
\]

Experimental FreeSolv values remain external validation only; no learned
element/class correction, linear calibration, or experimental residual is
allowed. Before any runtime promotion, the protocol requires all-\(3N\)
finite-difference and closed-loop conservative-force checks, dielectric and
dissociation limits, scaffold/transformation/charged holdouts, series ranking,
and a final dataset absent from teacher selection. No training data, model,
checkpoint, or `route1-physdistill` runtime has been created here.

- Mukhopadhyay et al., “Introducing Charge Hydration Asymmetry into the
  Generalized Born Model,” *J. Chem. Theory Comput.* (2014),
  DOI `10.1021/ct4010917`.
- Aguilar and Onufriev, “Efficient Computation of the Total Solvation Energy
  of Small Molecules via the R6 Generalized Born Model,”
  *J. Chem. Theory Comput.* (2012), DOI `10.1021/ct200786m`.
- Tan, Tan, and Luo, “Implicit nonpolar solvent models,”
  *J. Phys. Chem. B* (2007), DOI `10.1021/jp073399n`.

The sealed 2026-07-24/25 benchmark protocol JSON files retain an invalid
historical DOI string solely to preserve their cryptographic fingerprints.
That metadata is superseded by
`benchmarks/route1-citation-errata-2026-07-26.json`; it did not select a
formula, parameter, molecule, energy, or score.

## External higher-accuracy models screened but not integrated

IWM-GB augments a GBNSR6 dipolar base with higher water multipoles and
polar/nonpolar coupling evaluated on a NanoShaper solvent-excluded surface.
The 2024 paper reports test RMSEs of `0.95` and `0.87 kcal/mol` for its two
coupling variants on 85 rigid neutral H/C/N/O molecules. Those figures follow
parameter optimization against experimental hydration free energies on 88
training molecules; they are not label-independent confirmation on the full
FreeSolv domain. The source audit found no released conservative-force API, so
MAPLE does not expose IWM-GB for OPT or SCAN.

The OpenMM AGBNP source audit reached a different boundary. The reviewed
interface accepts per-atom radius, surface-tension, dispersion, charge, and
hydrogen flags and can return OpenMM forces, but the reviewed plugin implements
AGBNP1 and labels AGBNP2 as work in progress. A local Reference build against
OpenMM 8.5.2 produced a finite two-particle energy; this is an ABI smoke, not
small-molecule typing, force-parity, or hydration-accuracy evidence. The older
AGBNP3 plugin adds hydrogen-bond typing and an explicit bonded-neighbor graph,
with its example parameter path tied to Desmond DMS/OPLS. Route 1 therefore
does not add either dependency until a maintained, redistributable generic
small-molecule typing provider and a full conservative-force gate exist.

An exact CHA-GB-to-OpenMM shortcut is also unavailable. The CHA correction in
the published energy equation is analytical, but the audited GBNSR6 endpoint
computes effective Born radii from its R6 molecular-surface construction and
uses CHA-specific intrinsic radii. Encoding the charge-asymmetry factor on top
of OBC effective radii would define a new hybrid model, not reproduce the
frozen CHA-GB endpoint. Automatic differentiation of that hybrid would prove
only that the hybrid is conservative; it would not establish physical or
accuracy parity with CHA-GB/GBNSR6.

GBMV2 is a more direct physical candidate. Its Route 1 composition would be

\[
G_{\mathrm{solv}}(R)
=G_{\mathrm{GBMV2}}(R,q_{\mathrm{fixed}},r)
+\gamma\,\mathrm{SASA}(R)+\beta.
\]

The five-parameter analytical molecular-volume model estimates Born radii from
smooth atomic volume functions and provides coordinate derivatives; CHARMM
documents analytical method II as preferred for dynamics. Knight and Brooks
used GAFF/AM1-BCC, paired 10.5 ns vacuum and GBMV2 trajectories, and BAR for
499 neutral molecules. The reported optimized-profile AUE/RMSE was
`1.14/1.60 kcal/mol`, but the SASA coefficient was selected with experimental
hydration labels. A later comparison selected the GAFF/AM1-BCC coefficient on
82 CGENFF compounds and obtained AUE `1.24 kcal/mol` and \(R^2=0.758\) on
375 other compounds. These results motivate a provider audit rather than
proving improvement over the current MAPLE profile.

The usable upstream boundary is currently registered CHARMM/pyCHARMM. The
GPU implementation paper reports a CHARMM/OpenMM plugin and analytical
electrostatic plus SASA forces, but the audited public OpenMM tree contains no
GBMV2 force class. No local runtime was available to establish provider parity,
complete finite-difference forces, deployable license terms, or small-molecule
speed. Route 1 therefore does not reconstruct the algorithm from the paper.

GBSW and FACTS preserve the same additive boundary:

\[
G_{\mathrm{solv}}^{\mathrm{GBSW}}(R)
=G_{\mathrm{GBSW,polar}}(R,q_{\mathrm{fixed}},r)
+\gamma\,\mathrm{SASA}(R),
\]

\[
G_{\mathrm{solv}}^{\mathrm{FACTS}}(R)
=G_{\mathrm{FACTS,polar}}(R,q_{\mathrm{fixed}},r)
+\gamma\,\mathrm{SASA}_{\mathrm{FACTS}}(R).
\]

GBSW uses a smooth dielectric switching function and its official module
returns electrostatic-plus-nonpolar solvation energy and forces. Published
GAFF/AM1-BCC results are AUE/RMSE `1.20/1.52 kcal/mol` for 499 molecules and
AUE `1.33 kcal/mol` on a later 375-compound set separated from nonpolar
coefficient selection. Its documented cost is about four times slower than
vacuum; `2--3x` faster refers only to GBMV. FACTS analytically estimates
volume, spatial symmetry, Born radii, and SASA, but reports
`1.25/1.80 kcal/mol` AUE/RMSE and later AUE `1.42 kcal/mol`. Its original
speed is also about four times vacuum, and the official parameter set covers
protein radii only, using `TAVW` interpolation for unknown radii. Thus neither
model supports a faster-than-bare-MM claim.

CHARMM c50b2 names a GBSW OpenMM plugin in the registered CHARMM build.
Public OpenMM core contains neither GBSW nor FACTS, and no local runtime is
available for provider or force parity. The formulas remain valid scientific
candidates, but Route 1 adds no provider or paper-derived reimplementation.

SLIC/CDC is another physical additive candidate:

\[
G_{\mathrm{solv}}^{\mathrm{SLIC/CDC}}(R,q_{\mathrm{fixed}})
=G_{\mathrm{SLIC,es}}(R,q_{\mathrm{fixed}})
+G_{\mathrm{cav}}(R)
+G_{\mathrm{disp}}(R)
+G_{\mathrm{comb}}(R)
+G_{\mathrm{HB}}(R).
\]

The 2022 molecular study used the Mobley AM1-BCC structures/charges and reports
sub-kcal average aqueous accuracy, but the aqueous profile has 38 fitted
physical-model parameters. Its SI prose says 65 neutral training compounds,
whereas Table S4 lists 63. Independent layout/raw extraction of Table S10
produces the same 494 rows; recomputation gives MAE/RMSE
`0.813/1.152 kcal/mol`, and excluding the 63 listed training names leaves 431
historical rows at `0.826/1.190`. The `1.152` RMSE matches the ChemRxiv
preprint's `1.15`, not the SI footer's `0.98`; the table also does not reconcile
with the article's 500-solute or Appendix E's 502-corpus statements. The
historical holdout is not a prospective MAPLE reserve.

No complete upstream for the published model was identified. The public
Figshare record contains only a PDF. PBJ implements related SLIC
electrostatics and simple SASA nonpolar energy, but not the complete fitted
SLIC/CDC model; it has no complete atom-resolved polar-plus-CDC force. The
older MATLAB repository is an ion/solvent parameter-optimization code. Route 1
therefore records an energy-only watch item and does not reimplement the model
from its equations.

dSASA supplies analytical coordinate derivatives for an exact geometric SASA
term and is implemented in Amber. It remains a surface-area model rather than
the PBSA `inp=2` cavity-plus-dispersion endpoint that participated in the
lowest-error pairing. It therefore cannot be substituted silently for that
nonpolar term, and it does not provide the missing CHA-GB polar derivative.

### MLSES PB surface feasibility boundary

The tested AmberTools `sasopt=3, mlses_opt=0` path executes GENIUSES, which
changes PB dielectric-boundary construction rather than the Route 1 additive
identity. Its primary paper (DOI `10.1021/acs.jpclett.3c02176`) learns a
classical solvent-excluded-surface point-cloud geometry. The earlier MLSES
classifier (DOI `10.1021/acs.jctc.1c00492`) is its predecessor, not the
executed runtime. Neither model trains a hydration-energy residual or an
MLIP-specific molecular correction, so this class of surface surrogate is not
residual cheating in principle.

The maintained executable boundary is stricter. AmberTools 26 documents an
MLSES single-point example with `sasopt=3`, `mlses_opt=0`, `ipb=2`,
`eneopt=1`, and `frcopt=0`; its force example does not enable MLSES. A complete
runtime-input discovery over `ENEOPT=1..4` and `FRCOPT=1..5` under the fixed
linear-PB setup identifies five runtime-accepted force pairings:
`1/1`, `2/2`, `2/3`, `2/4`, and `3/2`. Classical SES writes nonempty atom
forces for all five at both grids, while GENIUSES writes no atom-resolved MLSES
force for any of them. Because no candidate force vector survives, the
required comparison

\[
F_i^{\mathrm{GENIUSES}}\stackrel{?}{=}
-\frac{G_{\mathrm{PB,GENIUSES}}(R+h e_i)
       -G_{\mathrm{PB,GENIUSES}}(R-h e_i)}{2h}
\]

could not be reached. On the same 23-atom CPU probe, the GENIUSES energy-only
path was also slower than classical SES at both grid settings. Consequently
there is no MLSES runtime provider and no derivative-based Route 1 task uses
this surface. This rejection is about present force and performance
capability, not the scientific legitimacy of a geometry surrogate.

GNNIS is excluded for a different, contractual reason rather than for a
missing derivative. The released implementation computes a scalar
implicit-solvent energy from GNN-scaled GB-Neck2 effective Born radii and a
learned SASA term, then obtains forces by automatic differentiation. Its
training labels are mean explicit-solvent solute forces after subtracting
vacuum OpenFF solute forces, and its published training code uses a force-only
mean-squared-error loss. The upstream vacuum force field is external to the
Torch solvent force, so the rejection is not an allegation of a hidden
gas-phase MM term. It is a pretrained chemistry-dependent solvent functional,
which violates Route 1's no-learned-correction baseline.

The 2025 QM-GNNIS transfer makes the distinction even more explicit:

\[
\Delta\Delta G_{\mathrm{corr}}
\approx G_{\mathrm{GNNIS}}-G_{\mathrm{GB\mbox{-}Neck2}},
\qquad
G_{\mathrm{QM\mbox{-}GNNIS}}
=E_{\mathrm{QM,CPCM}}+\Delta\Delta G_{\mathrm{corr}}.
\]

The paper and released delta class use this learned explicit-solvent
correction to improve relative conformer and spectroscopic predictions without
new QM/MM or experimental training. Those are useful scientific properties,
but they do not turn the term into a fixed physical PB/GB provider. Applying
the published QM transfer to a gas MLIP would be an additional hypothesis, not
evidence of multi-MLIP validation. The released workflow actually combines the
Torch delta with ORCA/CPCM through an ASE sum calculator, and the paper frames
the result as an emulation of QM/MM with electrostatic embedding and a
nonpolarizable MM solvent. It is therefore not a direct fixed-charge
MLIP-plus-PB/GB provider. Route 1 records the source audit without adding a
dependency, provider, FreeSolv selector, or runtime option.

AmberTorchPB modernizes the solution of a preassembled finite-difference PB
linear system with LibTorch. The source reviewed at commit
`a92c90b9e57726a9816de105892dd5b2ff2aae9c` exposes sparse-system solver
wrappers, not a molecular coordinate/radius/charge frontend or a coordinate
force. Its repository also contains no license file. It is an upstream watch
item, not a current redistributable Route 1 provider.

- Tolokh et al., “Implicit Water Multipole Generalized Born Model,”
  *J. Phys. Chem. B* (2024), DOI `10.1021/acs.jpcb.4c00254`.
- Gallicchio, Paris, and Levy, “The AGBNP2 implicit solvation model,”
  *J. Chem. Theory Comput.* (2009), DOI `10.1021/ct900234u`.
- Knight and Brooks, “Surveying implicit solvent models for estimating small
  molecule absolute hydration free energies,” *J. Comput. Chem.* (2011), DOI
  `10.1002/jcc.21876`.
- Vanommeslaeghe et al., “Assessing the quality of absolute hydration free
  energies among CHARMM-compatible ligand parameterization schemes,”
  *J. Comput. Chem.* (2013), DOI `10.1002/jcc.23199`.
- Gong et al., “Accelerating the Generalized Born with Molecular Volume and
  Solvent Accessible Surface Area Implicit Solvent Model Using Graphics
  Processing Units,” *J. Comput. Chem.* (2020), DOI `10.1002/jcc.26133`.
- Im, Lee, and Brooks, “Generalized Born Model with a Simple Smoothing
  Function,” *J. Comput. Chem.* (2003), DOI `10.1002/jcc.10321`.
- Haberthür and Caflisch, “FACTS: Fast Analytical Continuum Treatment of
  Solvation,” *J. Comput. Chem.* (2008), DOI `10.1002/jcc.20832`.
- Mehdizadeh Rahimi et al., “Solvation Thermodynamics of Solutes in Water and
  Ionic Liquids Using the Multiscale Solvation-Layer Interface Condition
  Continuum Model,” *J. Chem. Theory Comput.* (2022), DOI
  `10.1021/acs.jctc.2c00248`.
- Mobley et al., “Small Molecule Hydration Free Energies in Explicit Solvent:
  An Extensive Test of Fixed-Charge Atomistic Simulations,”
  *J. Chem. Theory Comput.* (2009), DOI `10.1021/ct800409d`.
- PBJ related SLIC implementation:
  <https://github.com/bem4solvation/pbj>.
- Cao et al., “Exact Analytical Algorithm for the Solvent-Accessible Surface
  Area and Derivatives in Implicit Solvent Molecular Simulations on GPUs,”
  *J. Chem. Theory Comput.* (2024), DOI `10.1021/acs.jctc.3c01366`.
- Katzberger and Riniker, “A general graph neural network based implicit
  solvation model for organic molecules in water,” *Chemical Science* (2024),
  DOI `10.1039/D4SC02432J`;
  <https://github.com/rinikerlab/GNNImplicitSolvent>.
- Katzberger, Pultar, and Riniker, “Transferring Knowledge from MM to QM: A
  Graph Neural Network-Based Implicit Solvent Model for Small Organic
  Molecules,” *J. Chem. Theory Comput.* (2025), DOI
  `10.1021/acs.jctc.5c00728`;
  <https://github.com/rinikerlab/QM-GNNIS>.
- Wu et al., “AmberTorchPB: A Unified Framework for Poisson-Boltzmann-Based
  Reaction Field Energy Calculation via Tensor Computation,”
  *J. Chem. Theory Comput.* (2026), DOI `10.1021/acs.jctc.6c00085`;
  <https://github.com/yxwu21/AmberTorchPB>.
- OpenMM AGBNP plugin:
  <https://github.com/Gallicchio-Lab/openmm_agbnp_plugin>.
- Frozen local/source audit:
  `benchmarks/route1-provider-feasibility-2026-07-24.json`.
- Frozen learned-solvent boundary audit:
  `benchmarks/route1-learned-solvent-boundary-audit-2026-07-25.json`.
- Frozen GBMV2 feasibility audit:
  `benchmarks/route1-gbmv2-feasibility-audit-2026-07-25.json`.
- Frozen FACTS/GBSW feasibility audit:
  `benchmarks/route1-facts-gbsw-feasibility-audit-2026-07-25.json`.
- Frozen SLIC/CDC feasibility audit:
  `benchmarks/route1-slic-cdc-feasibility-audit-2026-07-25.json`.
- Source-bound SLIC/CDC SI Table S10 replay:
  `benchmarks/route1-slic-cdc-si-table-replay-2026-07-25.json`, generated by
  `benchmarks/replay_slic_cdc_si_table.py` from the SHA-256-pinned published
  SI PDF. Table S4 is parsed from the layout-preserving extraction; Table S10
  is cross-checked between `pdftotext -layout` and `-raw` extraction modes.

## No-fit multi-solvent 3D-RISM research gate (not integrated)

The product Route 1 contract remains the fixed-charge PB/GB composition at the
top of this ledger.  A water GB calculation cannot be made into a physically
defined arbitrary-solvent model merely by changing its outer dielectric.
OpenMM exposes that dielectric in the polar GB term, but its surface-area term
does not acquire a solvent molecular structure, density, dispersion response,
or hydrogen-bond response from \(\epsilon_s\).  Therefore no non-water
`#solv` option, no custom-solvent public API, and no silent water-parameter
reuse is introduced by this research gate.

The first label-free mathematical candidate worth testing is a **separate,
benchmark-only 3D-RISM solvent functional**, not a term to bolt onto the
current PB/GB implementation.  For solvent \(s\), solute geometry \(R\), and
solvent site \(\gamma\), write the site distribution as

\[
h_{\gamma,s}(\mathbf r;R)
=\sum_\alpha
\left(c_{\alpha,s}(\cdot;R)*\chi_{\alpha\gamma,s}\right)(\mathbf r),
\]

where \(c\) is the solute--solvent direct correlation function and
\(\chi_s\) is the solvent's site--site susceptibility.  With

\[
d_{\gamma,s}(\mathbf r;R)
=-\beta u_{\gamma,s}(\mathbf r;R)
 +h_{\gamma,s}(\mathbf r;R)-c_{\gamma,s}(\mathbf r;R),
\]

the Kovalenko--Hirata closure is

\[
g_{\gamma,s}(\mathbf r;R)=
\begin{cases}
\exp(d_{\gamma,s}), & d_{\gamma,s}\leq0,\\
1+d_{\gamma,s}, & d_{\gamma,s}>0.
\end{cases}
\]

The solver then defines a fixed-geometry excess chemical potential

\[
W_s^{\mathrm{RISM}}(R)
=\mu_{\mathrm{3D\mbox{-}RISM}}^{\mathrm{ex}}
\left[u_s(R),\chi_s,\text{closure}\right].
\]

If, and only if, a future independent provider passes all numerical and
scientific gates, its ensemble calculation must retain the gas MLIP rather
than introduce a gas-phase MM energy:

\[
U_{g}(R)=E_{\mathrm{MLIP,gas}}(R),\qquad
U_{s}(R)=E_{\mathrm{MLIP,gas}}(R)+W_s^{\mathrm{RISM}}(R),
\]

\[
Z_x=\int_\Omega \exp[-\beta U_x(R)]\,dR,\qquad
\Delta G_{s,\mathrm{conf}}
=-\beta^{-1}\ln\frac{Z_s}{Z_g}.
\]

The final \(1\,\mathrm M\) ideal-gas to \(1\,\mathrm M\) ideal-solution
standard-state mapping is a separately declared thermodynamic step.  It must
be bound to the MNSol convention and the provider's own convention before any
comparison; a fixed-geometry \(\mu^{\mathrm{ex}}\), an implicit-program
default, and an ensemble \(\Delta G\) are not interchangeable.

For a neutral solute at infinite dilution, the concentration-standard-state
mapping of a genuine excess chemical potential follows directly from

\[
\mu_{\mathrm{soln}}(C)
=k_{\mathrm B}T\ln\!\left(
\frac{C\Lambda^3}{q_{\mathrm{int}}}
\right)+\mu^{\mathrm{ex}},
\qquad
\mu_{\mathrm{gas}}^{\mathrm{id}}(C)
=k_{\mathrm B}T\ln\!\left(
\frac{C\Lambda^3}{q_{\mathrm{int}}}
\right).
\]

At equal standard concentrations, the ideal translational and internal terms
therefore cancel:

\[
\Delta G^*_{1\,\mathrm M\rightarrow1\,\mathrm M}
=\mu_{\mathrm{soln}}(C^\circ)-\mu_{\mathrm{gas}}^{\mathrm{id}}(C^\circ)
=\mu^{\mathrm{ex}},
\qquad C^\circ=1\,\mathrm{mol\,L^{-1}}.
\]

Thus a neutral 3D-RISM excess-chemical-potential quantity has a **zero**
concentration-standard-state shift before comparison to MNSol's Ben-Naim
quantity.  This statement does not select raw KH, GF, PC, or PC+: those remain
different approximations to the solvent excess work.  It also does not cover
single-ion conventions, surface/Galvani potentials, a
fixed-geometry-to-ensemble approximation, or a temperature mismatch.

MNSol's starred quantity is explicitly the Ben-Naim
\(1\,\mathrm M\) gas to \(1\,\mathrm M\) solution standard state.  If
\(\Delta G^\circ_{1\,\mathrm{atm}\rightarrow1\,\mathrm M}\) denotes a
provider quantity using a \(1\,\mathrm{atm}\) ideal-gas reference, then

\[
\Delta G^*_{1\,\mathrm M\rightarrow1\,\mathrm M}
=
\Delta G^\circ_{1\,\mathrm{atm}\rightarrow1\,\mathrm M}
-RT\ln\!\left(\frac{C^\circ RT}{P^\circ}\right).
\]

At \(298\,\mathrm K\), \(C^\circ=1\,\mathrm{mol\,L^{-1}}\), and
\(P^\circ=1\,\mathrm{atm}\), the dimensionless ratio is approximately
\(24.46\), so the subtracted term is approximately
\(1.89\,\mathrm{kcal\,mol^{-1}}\).  Conversely, no \(1.89\) term is applied to
the neutral excess chemical potential above because the ideal terms cancel at
equal concentration.  A future scoring protocol must still bind the exact
Amber output field, sign, closure/pressure convention, temperature, and
solute ensemble before opening MNSol labels.

Amber's names also separate three mathematical layers that must not be
collapsed.  `rism_excessChemicalPotential` is the closure functional (KH in
the present pilots), while the Gaussian-fluctuation value is a distinct
alternative functional.  PC/PC+ are pressure/ensemble corrections involving
bulk density and partial molar volume; the \(RT\ln(24.46)\) term is instead a
gas-standard-state conversion.  Neither the presence of PC+ nor the string
``excess chemical potential`` proves that a \(1\,\mathrm{atm}\) reference has
already been converted to the MNSol \(1\,\mathrm M\) convention.  Therefore
Route 1 keeps all four quantities separately named and applies no universal
\(\pm1.89\,\mathrm{kcal\,mol^{-1}}\) rule without a source-level convention
binding.

Pressure corrections are also mathematical model choices, not free offsets.
For a predeclared 3D-RISM pressure \(P_s^{\mathrm{RISM}}\), ideal pressure
\(P_s^{\mathrm{id}}\), and partial molar volume \(\bar V_s\), the common
forms are

\[
W_s^{\mathrm{PC}}
=W_s^{\mathrm{closure}}-P_s^{\mathrm{RISM}}\bar V_s,
\qquad
W_s^{\mathrm{PC+}}
=W_s^{\mathrm{PC}}+P_s^{\mathrm{id}}\bar V_s.
\]

The closure, grid, buffer, temperature, site model, susceptibility/XVV asset
hash, and exactly one of the raw/PC/PC+ conventions must be frozen before any
experimental values are opened.  Per-molecule correction selection, a global
intercept, a fitted universal correction, and a solvent-specific error offset
are prohibited.  Amber exposes PC+ directly in the present executable but not
the rigorous PC endpoint as an independently named output; that implementation
fact is not permission to promote PC+ after inspecting labels.

MNSol declares \(298\,\mathrm K\).  The existing methanol and ethanol
research assets are at \(298.15\,\mathrm K\), so they remain numerical-only
assets and are not eligible for an MNSol accuracy score.  An accuracy protocol
would first need a source-bound \(298.00\,\mathrm K\) susceptibility, including
density and dielectric at that state, or a prospectively derived temperature
treatment.  The \(0.15\,\mathrm K\) difference cannot be excused after seeing
an error.

The required asset for every solvent is consequently not a scalar setting but

\[
\mathcal A_s=
\bigl(
\text{site topology}_s,q_s,\mathrm{LJ}_s,T_s,\rho_s,
\epsilon_s,\text{1D closure}_s,\chi_s
\bigr),
\]

with a content hash for the final XVV file and every source input.  A change in
temperature, pressure/density, molecular model, or 1D-RISM closure changes
\(\mathcal A_s\) and requires a new susceptibility; reusing a water XVV for a
named organic solvent is a category error, not a conservative approximation.
The inspected local AmberTools 26 installation contains only SPC/cSPCE water
XVV assets (including a cSPCE/NaCl mixture), not solvent-specific assets for
the other fourteen MNSol panel solvents.  This is a local availability gate,
not a statement that AmberTools cannot in principle accept another XVV: it
means the present checkout must fail closed rather than call those solvents
supported.

The executable source-level form of this gate is
`benchmarks/route1_custom_solvent_asset_contract_v1.json` plus
`benchmarks/route1_custom_solvent_asset_audit.py`.  It accepts a prospective
single-solvent manifest only when its molecular site types and coordinates,
MDL, 1D-RISM input, and final XVV agree.  In particular, it checks the
Amber file conventions before the reduced-unit relations.  For a manifest
charge \(q_i^{(e)}\) in electron-charge units, the MDL field is not an
arbitrary proportional vector:

\[
q_i^{\mathrm{MDL}}=18.2223\,q_i^{(e)}.
\]

Amber's MDL flag is named `LJSIGMA`, but the source reads it as the per-site
\(R_{\min}/2\), doubles it internally, and uses the pair potential

\[
u_{ij}^{\mathrm{LJ}}(r)
=\epsilon_{ij}
\left[
\left(\frac{R_{\min,ij}}{r}\right)^{12}
-2\left(\frac{R_{\min,ij}}{r}\right)^6
\right],
\qquad
R_{\min,ij}
=\frac{R_{\min,i}}2+\frac{R_{\min,j}}2.
\]

Therefore a source parameter using the conventional
\(4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]\) form must be converted exactly as

\[
\left(\frac{R_{\min,i}}2\right)_{\mathrm{MDL}}
=2^{-5/6}\sigma_i,
\]

whereas a source already reporting Amber \(R_{\min}/2\) is copied without that
conversion.  Treating conventional \(\sigma\) as the MDL value would change
the potential and is rejected as a unit error.  After those mappings, the XVV
audit checks the declared-temperature relations

\[
q^{\mathrm{XVV}}_i
=\frac{q^{\mathrm{MDL}}_i}{\sqrt{k_{\mathrm B}T}},
\qquad
\epsilon^{\mathrm{XVV}}_i
=\frac{\epsilon^{\mathrm{MDL}}_i}{k_{\mathrm B}T},
\]

as well as site multiplicities, **indexed site identity**, proper-rotation
rigid geometry, site/species density, dielectric, closure, grid, and content
hashes.  The complete `grid_points * site_type_count^2` XVV correlation array
must have exactly the declared length and every token must parse as a finite
number.  The documented `g/cm3` and `kg/m3` density conversions include the
required factor of \(1000\) between mass-density and molar units.  This makes a
water XVV renamed as an organic solvent, a temperature-mismatched XVV, a
site-type coordinate swap, or a malformed/extended correlation payload fail
before experimental scoring.  The small rounding tolerance in that audit is
limited to independently rounded text MDL/XVV quantities and the documented
`rism1d` density-unit conversion; it is not a fitted thermodynamic parameter.
A merely proportional but non-\(18.2223\) MDL charge vector now fails closed.
A passing asset audit says only that one physical *input* has internally
consistent metadata and a structurally valid finite payload.  When a local
generation log and sealed evidence record are supplied, the manifest and
primary audit artifact cross-hash both; that binding still does **not** prove
executable replay or regeneration parity.  It does not make a solvent
provider, establish a standard-state conversion, or imply numerical
convergence, force availability, accuracy, or ranking utility.

### Literature disposition: what the nonaqueous 3D-RISM evidence does and does not buy

The 3D-RISM path remains worth a **prospective physical** test because the KH
functional supplies a molecular solvent structure and an analytical excess
chemical potential rather than a FreeSolv/MNSol residual.  It is not, however,
evidence that a generic 15-solvent, all-record sub-(1) kcal/mol Route 1
method already exists.  The published evidence must stay bound to its exact
solvent model, correction convention, source panel, and metric:

| source / endpoint | relevant published result | admissible inference for Route 1 |
| --- | --- | --- |
| Misin, Palmer, and Fedorov (2016), corresponding-states one-site 3D-RISM/PC+ | Its Table 3 reports RMSEs of 2.23 (acetonitrile, (N=7)), 1.86 (chloroform, (N=107)), 2.21 (diethyl ether, (N=70)), 2.65 (DMSO, (N=7)), 3.02 (ethyl acetate, (N=22)), and 2.24 kcal/mol (octanol, (N=245)). | A coarse nonpolar/corresponding-states construction is not a shortcut to the requested polar/protic multi-solvent accuracy, even though its parameter source is not an experimental solvation residual. |
| Sergiievskyi *et al.* 3D-RISM pressure analysis | The pressure correction PC has a thermodynamic derivation; the extra PC+ ideal-gas term is explicitly described as ad hoc at molecular solute size. | Raw/PC and PC+ are distinct frozen model conventions. PC+ cannot be silently rebranded as an entirely first-principles correction merely because it has no per-solute residual at runtime. |
| Published all-site 3D-RISM-KH studies in individual solvents | Some reports give much smaller errors for selected solvents/models, for example chloroform or octanol. | These are useful feasibility evidence only. Their force fields, closures/corrections, source records, geometry/standard-state policies, and MAE versus RMSE definitions differ; their numbers must not be pooled or ranked against the Misin table or MAPLE artifacts. |

The practical disposition is therefore deliberately narrow:

1. the only currently admissible **new** physical research comparator is an
   all-site molecular DRISM/3D-RISM-KH calculation with a complete, frozen
   asset bundle \(\mathcal A_s\), using one predeclared raw or PC convention;
2. PC+ may be retained as a separately labeled benchmark comparator, never a
   hidden accuracy correction or a selector tuned after MNSol labels; and
3. the coarse one-site published assets, universal correction, SMD descriptor
   terms, multi-solvent regression models, and fitted PBSA surface profiles do
   not satisfy the requested no-residual physical route.

Most importantly, this comparator does **not** satisfy or redefine the Route 1
product formula.  A full 3D-RISM excess functional is not evidence for the
existing PB/GB decomposition \(G_{\mathrm{polar}}+G_{\mathrm{nonpolar}}\).
Its numerical or prospective accuracy evidence must remain benchmark-only
unless a future project explicitly adopts a different route contract.

Before that candidate receives any accuracy claim, each solvent must first pass
the asset contract, a 1D-RISM self-consistency/convergence check, 3D-RISM grid
and closure convergence at fixed geometry, standard-state closure, and a
label-blind energy run.  Only then may the separate scorer inspect the frozen
per-solvent MNSol labels.  A source-wide mean or an impressive result in one
solvent cannot replace the required per-record maximum-error and independent
external-test gates.

The executable falsification boundary is frozen in
`benchmarks/route1_multisolvent_accuracy_contract_v1.json`.  It content-binds
the 15-solvent/1,106-record label-free source audit and the single-solvent
physical-asset contract.  Its first complete-panel milestone is
\(\max_i|\widehat{\Delta G_i}-\Delta G_i|<1.5\) kcal/mol over every one of the
1,106 records; the stronger internal target is the same maximum below
\(1.0\) kcal/mol.  These are strict per-record conditions, not pooled-error
targets.  The final external test must be sourced and hash-frozen without
MNSol/FreeSolv record overlap or prior use, then evaluated once with the same
complete-denominator \(<1.5\) kcal/mol condition.  At present none of the
required 15-solvent asset set, sealed energy artifact, score, or external
protocol exists.  The contract also records that 3D-RISM cannot satisfy the
fixed PB/GB Route 1 product accuracy gate, so the equations above remain a
research hypothesis rather than a Route 1 accuracy result or runtime
capability.

The first label-free numerical pilot is frozen separately in
`benchmarks/route1_3drism_single_solvent_pilot_protocol_v1.json` and
`benchmarks/route1-3drism-single-solvent-pilot-2026-07-29.json`.  It uses the
AmberTools cSPCE molecular model and the `cSPCE_pse3.xvv` whose pinned 1D input
actually declares DRISM/PSE3; it does not substitute the separately named KH
XVV without an equally pinned generation input.  The 3D closure is
prospectively fixed to KH, and the primary quantity is the uncorrected
\(\mu^{\mathrm{ex}}_{\mathrm{KH}}\).  GF and PC+ are diagnostic outputs only.
For one fixed methyl-hexanoate geometry, changing `0.30 -> 0.35 A` grid spacing,
`14 -> 12 A` buffer, or `1e-5 -> 1e-4` solver tolerance changes the primary
quantity by `0.006905`, `0.000434`, and `0.001754 kcal/mol`, respectively.
An independent repeat at the reference settings differs by
`7.11e-13 kcal/mol`.  The sealed artifact records both the byte-for-byte
SHA-256 and a reproducibility SHA-256 for every generated topology file.
For `molecule.prmtop` only, the latter replaces the single wall-clock
`%VERSION ... DATE` field with a declared sentinel before hashing
(`amber_prmtop_version_date_header_v1`); every molecular-topology byte remains
bound, and the raw hash is retained as provenance.  These results qualify one
asset/solver combination
numerically; they do not apply the outstanding standard-state conversion,
inspect a hydration label, establish accuracy, supply forces, or authorize a
water/non-water product endpoint.

### First qualified non-water numerical candidate: three-site methanol

The first non-water asset to pass the same label-free numerical gate is the
three-site H/O/CH3 methanol model tabulated in the official ADF 3D-RISM
documentation.  ADF states that its solvent table is compiled from the
literature; MAPLE does not depend on or execute ADF.  The published site
parameters are independently re-expressed in the audited Amber MDL convention:

| site | \(q/e\) | conventional \(\sigma\) (A) | \(\epsilon\) (kcal/mol) | coordinate (A) |
| --- | ---: | ---: | ---: | --- |
| H | +0.435 | 1.000 | 0.056 | \((0,0,0)\) |
| O | -0.700 | 3.070 | 0.170 | \((0,0,0.945)\) |
| CH3 | +0.265 | 3.775 | 0.207 | \((1.35136,0,1.39716)\) |

The committed MDL therefore uses \(18.2223q/e\) for `CHG` and
\(2^{-5/6}\sigma\) for `LJSIGMA`.  The bulk state is frozen independently of
any solvation label at \(T=298.15\,\mathrm K\), \(24.550\,\mathrm{mol\,L^{-1}}\),
and \(\epsilon_r=32.63\).  The molar density is the NIST equation-of-state value
at 298.15 K; the dielectric is the NBS 25-degree-Celsius recommended value.
AmberTools `rism1d` with DRISM/KH, 4096 radial points, \(0.025\,\AA\) spacing,
and a \(10^{-8}\) residual threshold converges the primary RISM solve in
90 iterations at \(4.1372\times10^{-9}\).  Its subsequent temperature-
derivative solve converges in 66 iterations at \(4.8501\times10^{-9}\);
the committed generation log and sealed cross-hash artifact retain both
records rather than conflating the derivative solve with the primary one.

The immutable inputs and susceptibility are stored under
`benchmarks/route1_3drism_assets/methanol-adf3/`; the protocol and sealed
fixed-geometry result are
`benchmarks/route1_3drism_methanol_pilot_protocol_v1.json` and
`benchmarks/route1-3drism-methanol-pilot-2026-07-29.json`.  The numerical
thresholds were copied unchanged from the water pilot before this 3D result
was run.  Relative to the `0.30 A`, `14 A`, `1e-5` raw-KH reference, the
predeclared probes give:

| probe | \(\left|\Delta\mu^{\mathrm{ex}}_{\mathrm{KH}}\right|\) (kcal/mol) | frozen gate (kcal/mol) |
| --- | ---: | ---: |
| `0.35 A` grid | 0.004939 | 0.020 |
| `12 A` buffer | 0.001124 | 0.010 |
| `1e-4` tolerance | 0.000437 | 0.005 |
| independent reference repeat | \(2.98\times10^{-14}\) | \(1.0\times10^{-8}\) |

All five cases converge, so this is the first result with status
`one_nonwater_solver_pilot_numerically_qualified_not_accuracy_validated`.
The reference raw-KH value, `3.212676996 kcal/mol`, is a solver output, not an
accuracy score or yet an MNSol-comparable value: the raw functional's
standard-state mapping remains unproved, no MNSol value has been opened, and
neither PC nor PC+ has been selected.  This result counts as one **numerically qualified
research asset**, not one accuracy-qualified solvent and not one production
custom-solvent option.

This success is model-specific rather than evidence that any methanol topology
will converge.  A separate source-complete all-atom Amber methanol model based
on the bundled Caldwell--Kollman files failed the bounded DRISM/KH matrix:
residuals after 1,000 steps remained `7.2166`, `101.149`, `4.3361`, and
`0.5645` over the two tested radial grids and MDIIS step widths.  It was not
promoted or substituted after failure.  The accepted three-site asset is
therefore selected by provenance plus prospective numerical convergence, not
by an experimental error comparison.

### Second qualified non-water numerical candidate: four-site united-atom ethanol

The second non-water asset uses the H/O/CH2/CH3 united-atom ethanol model in
Table 7 of the official ADF 3D-RISM documentation:

| site | \(q/e\) | conventional \(\sigma\) (A) | \(\epsilon\) (kcal/mol) | coordinate (A) |
| --- | ---: | ---: | ---: | --- |
| H | +0.435 | 1.000 | 0.056 | \((0,0,0)\) |
| O | -0.700 | 3.070 | 0.170 | \((0,0,0.945)\) |
| CH2 | +0.265 | 3.775 | 0.207 | \((1.356103,0,1.398746)\) |
| CH3 | 0.000 | 3.905 | 0.175 | \((1.342751,0,2.928687)\) |

The same audited translations \(q_{\mathrm{MDL}}=18.2223q/e\) and
\((R_{\min}/2)_{\mathrm{MDL}}=2^{-5/6}\sigma\) are applied.  The thermodynamic
state is independently sourced at \(298.15\,\mathrm K\): Calvar *et al.*
report \(0.78546\,\mathrm{g\,cm^{-3}}\), which with the conventional ethanol
molar mass \(46.06844\,\mathrm{g\,mol^{-1}}\) gives
\(17.049850\,\mathrm{mol\,L^{-1}}\), rounded to the committed `17.0499 M`
input.  The explicit MDL site masses instead sum to `46.068`; they define the
site model and follow the pinned AmberTools `parm10.dat` convention
(`C=12.01`, `H=1.008`, `O=16.00`), but are not the mass used for that
concentration rounding.  The ADF table's separate `Weight=47.07` header is
therefore documented but not used.  NIST ThermoML records the committed static relative permittivity
\(\epsilon_r=24.35\) at \(298.15\,\mathrm K\) and \(100\,\mathrm{kPa}\).

AmberTools DRISM/KH with 4096 radial points, \(0.025\,\AA\) spacing, and a
\(10^{-8}\) threshold converges the primary 1D solve in 137 iterations at
\(9.5736\times10^{-9}\); the temperature-derivative solve converges in 79
iterations at \(9.5696\times10^{-9}\).  The immutable inputs, XVV, full log,
and cross-hash evidence are stored under
`benchmarks/route1_3drism_assets/ethanol-adf4/`.  The frozen protocol and
fixed-geometry result are
`benchmarks/route1_3drism_ethanol_pilot_protocol_v1.json` and
`benchmarks/route1-3drism-ethanol-pilot-2026-07-29.json`.

| probe | \(\left|\Delta\mu^{\mathrm{ex}}_{\mathrm{KH}}\right|\) (kcal/mol) | frozen gate (kcal/mol) |
| --- | ---: | ---: |
| `0.35 A` grid | 0.001464 | 0.020 |
| `12 A` buffer | 0.000645 | 0.010 |
| `1e-4` tolerance | 0.000570 | 0.005 |
| independent reference repeat | \(2.44\times10^{-14}\) | \(1.0\times10^{-8}\) |

All five cases converge.  The raw-KH reference value,
`-2.681003137 kcal/mol`, remains an unconverted solver output.  Ethanol is a
member of the frozen 15-solvent target panel, but no ethanol solvation label
was opened and this result supplies no standard-state, accuracy, force,
runtime, ranking, or endpoint-selection evidence.  It is therefore a second
**numerically qualified research asset**, not an accuracy-qualified panel
solvent.

### Failed label-free thermodynamic-consistency qualification

The next prospective experiment was frozen before execution in
`benchmarks/route1_3drism_thermodynamic_consistency_protocol_v1.json`, with
result
`benchmarks/route1-3drism-ethanol-thermodynamic-consistency-2026-07-29.json`.
It neither reads an experimental solvation value nor changes the Route 1
product formula.

Rigid translation and bonded-parameter independence pass at roundoff scale.
For the latter, all bond, angle, and dihedral force constants are multiplied
by \(1.5\), while the charge vector, atom-type indices, and LJ \(A/B\) tables
are proven unchanged.  The maximum change among raw KH, GF, and PC+ is
\(1.95\times10^{-14}\,\mathrm{kcal\,mol^{-1}}\), and the PMV change is
\(7.39\times10^{-13}\,\AA^3\).  This is direct executable evidence that the
reported solvent quantities do not include the bonded MM energy.

The full qualification fails for two distinct reasons:

1. the proper rotation changes the automatically generated Cartesian domain
   from \(112\times108\times128\) to \(126\times120\times108\).  The raw-KH
   change is \(0.004713\), but the GF change is
   \(0.020285\,\mathrm{kcal\,mol^{-1}}\), above the predeclared \(0.01\)
   gate.  Because the quadrature domain changed, this is a failed numerical
   qualification, not evidence that the continuous scalar functional lacks
   rotational invariance.  A new test must freeze a common grid before
   execution rather than loosen this threshold afterward.
2. setting the prmtop charge and LJ \(A/B\) coefficients to zero does not
   create \(u_{uv}=0\) in Amber.  Its solute constructor explicitly maps zero
   \(B\) to \(\sigma=0.7\,\AA\) and zero \(A\) to
   \(\epsilon=10^{-2}\); see the pinned
   [AmberClassic source](https://github.com/Amber-MD/AmberClassic/blob/0b35bfeb96026ffa4e5876391a0828f39b3cfc8d/src/rism/rism3d_solute_c.F90#L167-L180).
   The observed raw-KH/GF values \(9.43298/5.78621\) therefore diagnose this
   fallback and cannot be interpreted as a zero-interaction-limit failure of
   the KH functional itself.

The correct mathematical zero-interaction requirement remains

\[
u_{uv}(\mathbf r)\equiv0
\Longrightarrow h_{uv}(\mathbf r)=c_{uv}(\mathbf r)=0
\Longrightarrow
\mu_{\mathrm{KH}}^{\mathrm{ex}}
=\mu_{\mathrm{GF}}^{\mathrm{ex}}=0.
\]

A future executable probe must predeclare an explicit solute- or pair-potential
override, verify the resulting tabulated \(u_{uv}\) directly, and use one fixed
grid for every rigid orientation.  This failed experiment does not authorize
endpoint switching, threshold relaxation, an accuracy score, or product
promotion; work returns to the CHA-GB/PBSA ranking mainline.

### Rejected first non-water candidate: three-site acetonitrile

The first prospective non-water candidate was deliberately investigated
without opening any solvation label.  The strongest open, executable source
found is the RISMiCal acetonitrile example at commit
`e337cdebecc2dcb3ae3dc16d939d0cb1422deea1` (MIT license).  It freezes a
three-site model at \(298\,\mathrm K\) and \(19.18\,\mathrm M\):

| site | \(q/e\) | conventional \(\sigma\) (A) | \(\epsilon\) (J/mol) | axial coordinate (A) |
| --- | ---: | ---: | ---: | ---: |
| N | -0.430 | 3.200 | 711.3 | +1.157 |
| C | +0.280 | 3.650 | 627.6 | 0.000 |
| CH3 | +0.150 | 3.775 | 866.1 | -1.458 |

Its supplied charge-up/KH calculation converges below \(10^{-8}\), but the
input does **not** enable DRISM: this is a plain 1D-RISM result and cannot be
reported as proof that the same model converges under Amber DRISM.  The
deterministic Amber translation used the fixed \(18.2223\) charge factor,
\(2^{-5/6}\sigma\) radii, and J/mol-to-kcal/mol epsilon conversion.  With the
chemistry and thermodynamic state frozen, the exploratory Amber DRISM/KH
results were:

- the water-like default `NR=16384`, `DR=0.025 A`, `MDIIS_DEL=0.3`,
  `tolerance=1e-12` entered a restart/stagnation state at residual
  \(5.8451\times10^{-1}\);
- `NR=2048`, `MDIIS_DEL=0.1` decayed smoothly but remained near
  \(1.7\times10^{-6}\) after 3,000 steps;
- changing `MDIIS_RESTART` from 10 to 100 or 1,000 did not remove that
  residual floor;
- tolerance homotopy converged at \(10^{-5}\), then failed the next
  \(10^{-6}\) stage at \(1.56294\times10^{-6}\); and
- cross-process charge continuation was unstable rather than reproducing
  RISMiCal's in-process plain-RISM charge-up.

The separate pyRISM six-site acetonitrile input is technically parseable as a
DRISM/KH model, but its exact site parameters have no explicit literature
provenance beside the checked-in TOML and the repository contains no
preconverged acetonitrile susceptibility.  It is therefore not substituted
after the three-site failure.

The scientific conclusion is narrow: neither acetonitrile candidate currently
satisfies both source provenance and Amber DRISM convergence.  No acetonitrile
XVV or pilot is committed and no label is opened.  The later methanol success
does not repair this solvent-specific failure or count acetonitrile toward the
15-solvent gate.  A future acetonitrile attempt must introduce new
source-complete evidence or a genuinely different, prospectively declared
solver/model route; merely loosening the residual threshold or relabeling
plain RISM as DRISM is prohibited.

There are also tempting but inadmissible shortcuts.  The 3D-RISM universal
correction is a regression
\(\Delta G_{\mathrm{UC}}=\Delta G_{\mathrm{GF}}+a\rho\bar V+b\) against
experimental values, so it violates the no-fit rule even when its reported
average error is small.  The cavity-corrected and learned/descriptive RISM
variants similarly require a separate provenance review before they could
enter a no-fit screen; they are not silently treated as PC or PC+.

Misin, Palmer, and Fedorov's corresponding-states PC+ construction is the
useful narrower counterexample: it estimates coarse-grained **nonpolar**
solvent Lennard--Jones parameters from critical-point properties and reports
non-aqueous 3D-RISM calculations without electrostatics.  The exact published
supporting archive has now been audited without opening its result tables in
`benchmarks/route1-misin-coarse-grain-asset-audit-2026-07-29.json`.  It names
only eight of the required fifteen solvents, and each allowlisted asset is an
HNC, 298.15 K, one-site/one-species model with zero site charge and
`epsilon_XVV=1.0`.  Its 1D generator inputs show `DIEps=2.0`, but the frozen
XVV assets themselves contain no molecular electrostatic or hydrogen-bond
sites.  The archive also contains 21 per-solute result CSV members, which the
audit detects by metadata only and never reads.

Thus the **corresponding-states idea** remains a literature lead for a
separately predeclared nonpolar-only investigation, but these published assets
are not a physical molecular solvent definition for the polar, protic, or
hydrogen-bonding members of the required panel.  They are not vendored,
installed, scored against MNSol, counted toward the 15-solvent gate, or mixed
with another endpoint and called a single all-solvent solution.  Nor can the
paper's average error be substituted for the user's per-record
maximum-error gate.

This explains the custom-solvent boundary precisely.  A physically specified
3D-RISM solvent needs molecular site positions, charges, Lennard--Jones
parameters, temperature, density, and the resulting \(\chi_s\) (or a
verified equivalent susceptibility asset), not merely a dielectric constant.
Conversely, the SMD family accepts bulk descriptors such as dielectric,
refractive index, surface tension, and hydrogen-bond acidity/basicity because
it is a distinct density/IEF-PCM plus CDS model with its own published
parameterization.  Mixing SMD descriptor terms into water-calibrated GB/SASA
would define an unvalidated hybrid and is excluded.  Likewise, the recent
27-organic-solvent PBSA study reports SASA parameters fitted to 1,246 MNSol
values; it is useful literature evidence that a multi-solvent benchmark is
necessary, but its fitted profile is ineligible under Route 1's no-fit rule.

The committed source audit
`benchmarks/route1-mnsol-multisolvent-source-audit-2026-07-29.json` freezes a
15-solvent, 1,106-neutral-record MNSol coverage gate without reading or
redistributing \(\Delta G_{\mathrm{solv}}\) values.  A later prospective
calculation must report each solvent separately, retain the same solvent set,
perform leave-one-solvent-out sensitivity only without label fitting, and meet
the user's strict maximum-absolute-error gate for every evaluated record.  It
must then be followed by a separately sourced external set that was absent
from source selection, method selection, fitting, and all prior scoring.  This
coverage artifact is not an accuracy result and does not establish a
non-water runtime provider.

-  3D-RISM implementation and AmberTools context: Case et al., *AmberTools*,
  *J. Chem. Inf. Model.* (2023), DOI `10.1021/acs.jcim.3c01153`.
- Pressure-correction derivation and convention analysis:
  <https://arxiv.org/abs/1511.04475>.
- Nonaqueous corresponding-states 3D-RISM/PC+ result table (not adopted as a
  general molecular solvent model): Misin, Palmer, and Fedorov, *J. Phys.
  Chem. B* (2016), DOI `10.1021/acs.jpcb.6b05352`, open manuscript
  <https://strathprints.strath.ac.uk/56683/1/Misin_etal_JPCB_2016_Predicting_solvation_free_energies_using_parameter_free.pdf>.
- 3D-RISM-KH solvent structure and analytic thermodynamic functional example:
  Roy and Kovalenko, *J. Phys. Chem. B* (2015), DOI
  `10.1021/acs.jpcb.5b01291`.
- Amber RISM asset/XVV workflow: AmberTools Users' Manual, RISM chapter,
  <https://ics.uci.edu/~sources/amber/AmberTools.pdf>.
- Amber MDL charge and Lennard--Jones conventions: AmberClassic source,
  `constants_rism.F90`, `solvmdl_c.F90`, and `rism1d_potential_c.F90` at
  commit `0b35bfeb96026ffa4e5876391a0828f39b3cfc8d`,
  <https://github.com/Amber-MD/AmberClassic>.
- Open acetonitrile plain-RISM source model and convergence record:
  RISMiCal `example/acetonitrile_vv` at commit
  `e337cdebecc2dcb3ae3dc16d939d0cb1422deea1`,
  <https://github.com/rismical-dev/rismical/tree/master/example/acetonitrile_vv>.
- pyRISM DRISM/XRISM implementation and unqualified acetonitrile input lead:
  <https://github.com/2AUK/pyRISM>.
- Literature-compiled three-site methanol parameters:
  ADF 3D-RISM documentation, Table 6,
  <https://www.scm.com/doc/ADF/Input/3D-RISM.html>.
- Methanol density at 298.15 K: Goodwin, *J. Phys. Chem. Ref. Data* 16
  (1987), <https://srd.nist.gov/jpcrdreprint/1.555786.pdf>.
- Methanol dielectric constant at 25 degrees Celsius: Maryott and Smith,
  NBS Circular 514,
  <https://nvlpubs.nist.gov/nistpubs/Legacy/circ/nbscircular514.pdf>.
- Literature-compiled four-site united-atom ethanol parameters:
  ADF 3D-RISM documentation, Table 7,
  <https://www.scm.com/doc/ADF/Input/3D-RISM.html>.
- Ethanol density at 298.15 K: Calvar *et al.*, *J. Chem. Eng. Data* 55
  (2010), DOI `10.1021/je900998f`.
- Ethanol static relative permittivity at 298.15 K and 100 kPa:
  Uosaki *et al.*, NIST ThermoML record for *J. Chem. Eng. Data* 51
  (2006), <https://trc.nist.gov/ThermoML/10.1021/je060248p.html>.
- Rejected bundled all-atom methanol model provenance: Caldwell and Kollman,
  *J. Phys. Chem.* (1995), DOI `10.1021/j100016a067`.
- Parameter-free nonpolar corresponding-states PC+ lead: Misin, Palmer, and
  Fedorov, *J. Phys. Chem. B* (2016), DOI `10.1021/acs.jpcb.6b05352`.
- OpenMM GBSAOBC dielectric definition:
  <https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html>.
- SMD density/IEF-PCM/CDS model: Marenich, Cramer, and Truhlar, *J. Phys.
  Chem. B* (2009), DOI `10.1021/jp810292n`.
- Multi-solvent PBSA parameter study (not adopted because it fits its SASA
  parameters): DOI `10.1021/acs.jpcb.6c00810`.
- MNSol source and standard-state manual:
  <https://comp.chem.umn.edu/mnsol/MNSol-v2012_Manual.pdf>.

## Atomic charge providers

- AM1-BCC is invoked through AmberTools Antechamber `-c bcc`; see Jakalian et
  al., “Fast, efficient generation of high-quality atomic charges. AM1-BCC
  model: II. Parameterization and validation,” *J. Comput. Chem.* (2002),
  DOI `10.1002/jcc.10128`.
- ABCG2 is invoked through AmberTools24+ Antechamber `-c abcg2`; see “ABCG2: A
  Milestone Charge Model for Accurate Solvation Free Energy Calculation,”
  *J. Chem. Theory Comput.* (2025), DOI `10.1021/acs.jctc.5c00038`.
- The same-model transfer boundary is documented by “Evaluation of the ABCG2
  Charge Model in Protein--Ligand Binding Free-Energy Calculations,”
  *J. Chem. Inf. Model.* (2025), DOI `10.1021/acs.jcim.5c02161`: stronger
  hydration/solvation behavior did not establish a general protein--ligand
  relative-ranking advantage. Route 1 therefore does not select a charge
  method from hydration performance alone.
- Charge assignment can also depend materially on the input conformer; see
  “Evaluating the Functional Importance of Conformer-Dependent Atomic Partial
  Charge Assignment,” *J. Comput. Chem.* (2025), DOI
  `10.1002/jcc.70112`. This motivates a future prospectively frozen
  same-record sensitivity study, not reinterpretation of the current
  single-vector reserve as charge-model uncertainty.
- Antechamber's finite-precision provider path can leave a small residual from
  the declared integer charge.  MAPLE no longer accepts a molecule-independent
  `0.01 e` allowance.  The runtime derives a per-molecule upper bound from the
  three-decimal SQM Mulliken precharge resolution, the actual final-MOL2 charge
  tokens, atom count, and floating-point slack.  Only residuals within that
  serialization bound are distributed uniformly; all raw tokens, half-widths,
  formulas, and repaired values are written to the normalization audit.  A
  provider token coarser than the verified AmberTools 26 four-decimal MOL2
  floor is rejected rather than being allowed to enlarge its own tolerance.
  Larger residuals fail closed.  Uniform distribution follows the published
  FESetup workflow, while the acceptance bound is MAPLE's narrower
  provider-serialization guard rather than a charge-model parameter.  See
  Loeffler et al., “FESetup: Automating Setup for Alchemical Free Energy
  Simulations,” *J. Chem. Inf. Model.* (2015), DOI
  `10.1021/acs.jcim.5b00368`.
- RESP and RESP2 are not regenerated by MAPLE route 1; their charges are read
  from MOL2 without modification. RESP: “A well-behaved electrostatic potential
  based method using charge restraints for deriving atomic charges: the RESP
  model,” DOI `10.1021/j100142a004`; RESP2: “Non-bonded force field model with
  advanced restrained electrostatic potential charges (RESP2),” DOI
  `10.1038/s42004-020-0291-4`.

## LPB

The APBS adapter follows the official solvation-energy construction: one LPBE
calculation at solvent dielectric 78.5, one reference calculation at dielectric
1.0, followed by `print elecEnergy solv - ref end`.  The APBS APOLAR block
provides the required nonpolar term.  The locked generic profile reduces the
APBS nonpolar equation

\[
G_{np}=\gamma A+pV+\bar\rho\sum_i\int u_i^{att}\theta\,dy
\]

to `gamma*A` by setting `p=0` and `bar(rho)=0`; `gamma=0.105 kJ mol^-1 A^-2`
is the documented APBS/iAPBS default used by the profile.

The canonical provider control is the official radius-3-A, charge-+1 Born ion.
For zero ionic strength its analytic polar transfer energy is

\[
\Delta_pG_{\mathrm{Born}}=
\frac{q^2}{8\pi\epsilon_0a}
\left(\frac{1}{\epsilon_{out}}-\frac{1}{\epsilon_{in}}\right).
\]

The official APBS settings (`97^3`, `0.33 A`, `pdie=1`, `sdie=78.54`, LPBE,
`srfm=mol`, `srad=1.4`, `chgm=spl2`) report `-229.59 kJ/mol`; the analytic
formula gives `-230.62 kJ/mol`.  MAPLE stores the PQR, input, stdout, stderr,
binary hash, and parsed value.  Neutral methanol and aniline controls additionally
measure grid sensitivity from `0.50` through approximately `0.1667 A` at an
approximately fixed 32-A physical grid length.

The frozen development-only cross-provider screen replaces the APBS APOLAR
component, not the polar equation:

\[
G_{\mathrm{screen}}
=G_{\mathrm{APBS,LPBE}}^{\mathrm{polar}}
+G_{\mathrm{OpenMM,OBCII}}^{\mathrm{ACE}}.
\]

The APBS APOLAR and OpenMM ACE terms are mutually exclusive in this sum.  ACE
is the nonpolar component already decomposed from the OpenMM OBC-II/ACE
continuum force; no OpenMM gas-phase MM energy enters.  All 526 APBS energies
were sealed before FreeSolv values were opened.  The resulting
`1.658/2.535 kcal/mol` MAE/RMSE misses the frozen material-gain threshold, and
the label-free 20-case grid comparison reaches `0.671 kcal/mol` maximum and
`0.455 kcal/mol` P90 changes.  It is therefore a rejected SP benchmark, not a
new allowed provider pairing.

The post hoc numerical follow-up does not use hydration targets.  On the same
20 preselected cases, `129^3 @ 0.25 A` to `161^3 @ 0.20 A` reduces the
maximum/P90 polar-energy differences to `0.175/0.148 kcal/mol`.  A new frozen
full-development endpoint at the `129^3` grid gives MAE/RMSE
`1.629/2.462 kcal/mol`; its paired MAE gain over OBC-II/ACE is
`0.131 kcal/mol` with interval `[0.050, 0.217]`.  This proves that a finer
molecular-surface grid can be numerically adequate, but the improvement still
misses the prospective `0.15 kcal/mol` materiality gate and does not justify
the slower, energy-only cross-provider runtime.

APBS's force interface does not differentiate that molecular-surface endpoint:
the pinned 3.4.1 runtime aborts `srfm=mol` with
`Forces *must* be calculated with spline-based surfaces!`. A separate SPL4
probe evaluates

\[
F_i^{\mathrm{APBS,SPL4}}
\stackrel{?}{=}
-\frac{G_{\mathrm{polar}}(R+h e_i)-G_{\mathrm{polar}}(R-h e_i)}{2h}
\]

for all 69 Cartesian components at two steps and two grids. The finer-grid
RMSE/maximum remain `0.199/1.103 kJ/mol/A`, and the coarse-to-fine force
RMSE/maximum are `0.483/2.579 kJ/mol/A`. APBS describes the SPL2 dielectric
surface as force-suitable only after substantial force-field
reparameterization and describes SPL4 as a higher-continuity surface with
similar characteristics. The existing generic mbondi2 profile has no such
SPL4 calibration.

The APBS APOLAR documentation defines nonpolar forces as derivatives of the
same \(\gamma A+pV+\rho WCA\) energy and exposes a finite-difference
displacement. For the locked \(\gamma A\)-only profile, however, the
calculation-block total force fails an independent centered energy difference
at the matching `0.05 A` step (RMSE/maximum
`1.114/4.871 kJ/mol/A`). Consequently neither APBS force path is part of the
Route 1 product.

- Im, Beglov, and Roux, “Continuum solvation model: computation of
  electrostatic forces from numerical solutions to the Poisson-Boltzmann
  equation,” *Computer Physics Communications* (1998), DOI
  `10.1016/S0010-4655(98)00016-2`.
- Nina, Im, and Roux, “Optimized atomic radii for protein continuum
  electrostatics solvation forces,” *Biophysical Chemistry* (1999), DOI
  `10.1016/S0301-4622(98)00236-1`.
- Baker et al., “Electrostatics of nanosystems: Application to microtubules and
  the ribosome,” *PNAS* (2001), DOI `10.1073/pnas.181342398`.
- Jurrus et al., “Improvements to the APBS biomolecular solvation software
  suite,” *Protein Science* (2018), DOI `10.1002/pro.3280`.
- Wagoner and Baker, “Assessing implicit models for nonpolar mean solvation
  forces: the importance of dispersion and volume terms,” *PNAS* (2006),
  DOI `10.1073/pnas.0600118103`.

## External ddPCM/ddLPB audit

ddX supplies domain-decomposition implementations of ddCOSMO, ddPCM, and
ddLPB for point multipoles inside a cavity defined by overlapping atom-centered
balls. For the Route 1 zero-ionic-strength water screen, ddPCM is the relevant
\(\kappa=0\) limit of ddLPB. A finite ionic-strength profile would need to
declare ddLPB and its screening parameter explicitly; it is not interchangeable
with the frozen zero-salt endpoint.

The audited additive endpoint is

\[
G_{\mathrm{screen}}(R,q_{\mathrm{AM1-BCC}})
=G_{\mathrm{ddPCM}}^{\mathrm{polar}}(R,q_{\mathrm{AM1-BCC}})
+G_{\mathrm{OpenMM,OBCII}}^{\mathrm{ACE}}(R).
\]

No ddX, OpenMM, or other gas-phase MM energy appears in this expression. The
ACE value is the same already-frozen nonpolar component used by the OBC-II
baseline.

For `pyddx 0.8.0`, the tested native coordinate arrays obey the numerical
identity

\[
g_{\mathrm{native}}=
g_{\mathrm{solvation\_force\_terms}}
+g_{\mathrm{multipole\_force\_terms}}
=\frac{\partial G_{\mathrm{ddPCM}}}{\partial R_{\mathrm{bohr}}}.
\]

The conventional Route 1 force is therefore

\[
F_{\mathrm{kJ\,mol^{-1}\,A^{-1}}}
=-g_{\mathrm{native}}
\left(\frac{\mathrm{bohr}}{\mathrm{A}}\right)
\left(\frac{\mathrm{kJ\,mol^{-1}}}{E_h}\right).
\]

Both native contributions are required; treating the API's `force_terms` name
as an already-negated force reverses the sign. The isolated methyl-hexanoate
probe validates this conversion against centered energy differences for all
69 Cartesian components. The mbondi2 van-der-Waals cavity gives
`0.000404/0.001504 kJ/mol/A` primary-step RMSE/maximum error, and its
net-force norm is `8.3e-14 kJ/mol/A`.

This derivative result does not certify the physical pairing. A separate
label-free 20-molecule convergence check passes, but the 526-molecule
development endpoint gives `1.782/2.881 kcal/mol` MAE/RMSE, slightly worse in
MAE and clearly worse in RMSE/outliers than OBC-II/ACE
(`1.760/2.537 kcal/mol`). The expanded mbondi2+1.4-A cavity is much worse.
The local ddPCM polar force call is also roughly `360x` slower than the frozen
warm OpenMM OBC-II/ACE correction. Consequently the equations and derivative
convention are retained only as an external audit; no `pyddx` dependency or
MAPLE provider is introduced.

- ddX theory and model definitions:
  <https://ddsolvation.github.io/ddX/md_docs_theory.html>.
- Stamm et al., domain-decomposition linearized Poisson-Boltzmann model:
  <https://arxiv.org/abs/1807.05384>.
- Herbst et al., analytical ddLPB forces:
  <https://arxiv.org/abs/2203.00552>.

The ABCG2-specific PBSA profile is not approximated.  Its defining reference is
Sun et al., “Development and test of highly accurate endpoint free energy
methods. 1: Evaluation of ABCG2 charge model on solvation free energy prediction
and optimization of atom radii suitable for more accurate solvation free energy
prediction by the PBSA method,” *J. Comput. Chem.* (2023), DOI
`10.1002/jcc.27089`; execution stays blocked until its exact optimized radii and
nonpolar parameter files are obtained from an authoritative, redistributable
source and pass human review.

## Upstream source snapshots reviewed

- Caltech `cheq` 0.5.1 source and parameter table at commit
  `f71eeb4ab3790557fb244a8f09fa60e255ee6274`, especially
  `src/shielding/gto.rs`, `src/solver/implementation.rs`, and
  `resources/qeq.data.toml`.
- Open Babel QEq source and data at commit
  `0e94434fa75c9f61095023e3c12e0d5f2ac035ff`, used to identify and reject the
  fixed-hydrogen shortcut rather than as MAPLE's current numerical profile.
- APBS input/output and APOLAR execution paths at commit
  `4613d0d547c3c71df8815dcb85e9e19abf61822c`, especially
  `examples/solv`, `examples/born`, and `src/routines.c`.
- `pyddx 0.8.0` PyPI source archive
  `31a1ddfe72105a0a6843ef2bcda76763cb99bab56d85d844c08e078156201f08`;
  the isolated build passed `src/test_pyddx.py`, `src/test_lpb_limits.py`,
  `src/test_multipoles.py`, and `src/test_gradients.py` (`9 passed`).
- OpenMM 8.5 `openmm.app.internal.customgbforces` implementations and their
  `getStandardParameters(topology)` assignments for all five GB models.
- AmberTorchPB commit
  `a92c90b9e57726a9816de105892dd5b2ff2aae9c`; its public wrapper and tests
  operate on preassembled linear systems and expose no molecular force API.

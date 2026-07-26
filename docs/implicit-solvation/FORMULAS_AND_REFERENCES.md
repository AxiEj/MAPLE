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

- Mukhopadhyay et al., “Charge hydration asymmetry: the basic principle and
  how to use it to test and improve water models,” *J. Chem. Theory Comput.*
  (2014), DOI `10.1021/ct4010917`.
- Aguilar et al., “On the Evaluation of the Generating Function R6 for the
  Computation of the Effective Born Radii,” *J. Chem. Theory Comput.* (2025),
  DOI `10.1021/acs.jctc.4c01471`.
- Tan, Tan, and Luo, “Implicit nonpolar solvent models,”
  *J. Phys. Chem. B* (2007), DOI `10.1021/jp073399n`.

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

## Atomic charge providers

- AM1-BCC is invoked through AmberTools Antechamber `-c bcc`; see Jakalian et
  al., “Fast, efficient generation of high-quality atomic charges. AM1-BCC
  model: II. Parameterization and validation,” *J. Comput. Chem.* (2002),
  DOI `10.1002/jcc.10128`.
- ABCG2 is invoked through AmberTools24+ Antechamber `-c abcg2`; see “ABCG2: A
  Milestone Charge Model for Accurate Solvation Free Energy Calculation,”
  *J. Chem. Theory Comput.* (2025), DOI `10.1021/acs.jctc.5c00038`.
- Antechamber's finite-precision text output can have a small residual from the
  declared integer charge.  MAPLE follows the published FESetup procedure and
  distributes only residuals no larger than `0.01 e` uniformly over all atoms,
  preserving equivalence while recording the provider values, correction, and
  used values in a normalization audit.  Larger residuals fail closed.  See
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

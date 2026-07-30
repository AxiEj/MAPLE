# Route-2 V0: permanent-reference identifiability gate

## Status and decision

This document records a fail-closed V0 gate.  It does not introduce a new
model, change a checkpoint, fit a parameter, choose a cavity, run a PCM
calculation, or use an experimental solvation value.

The frozen V0 atomic independent-particle response table gives a useful
neutral induced-response covariance, and the full-response KKT kernel now
proves reciprocal/passive induction from one scalar.  Neither result identifies
the permanent molecular electronic source.  Therefore neither V0-AIPR nor
V0-ADT may yet be attached to a physical continuum or an accuracy panel.

The machine-readable decision is
[route2-v0-permanent-reference-identifiability-audit-v1.json](benchmarks/route2-v0-permanent-reference-identifiability-audit-v1.json).

## 1. What the current response asset determines

At a fixed nuclear geometry, let \(C\succeq0\) be the frozen response
covariance in its declared neutral support and let \(B\) take response
coefficients to an exterior electrostatic potential.  The existing common
scalar correctly determines the induced tangent response:

\[
\delta c^*(f)=-C f.
\]

It is enough to test reciprocity, passivity, KKT stationarity, and source/dual
pairing of the induced response.  It does not say what the molecular
ground/reference density is.  In particular, the atomic independent-particle
table contains neutral transition densities.  Each mode integrates to zero.
Its span carries tangent redistributions of an already-existing electron
density, not the electron-number-bearing molecular density itself.

The nuclear electrostatic potential is known exactly from the nuclear
coordinates and charges.  What remains unidentified is the stationary
permanent electronic contribution that must be paired with it in the total
solute source.

## 2. Reference-shift theorem

For any admissible neutral reference shift \(a\) in the response support,
define

\[
F_a(c)=\frac12(c-a)^\mathsf T C^+(c-a).
\]

The minimizer in an external coefficient dual \(f\) is

\[
c_a^*(f)=a-Cf,
\qquad
\frac{\partial c_a^*}{\partial f}=-C.
\]

Thus every \(a\) produces exactly the same response covariance and satisfies
the same reciprocity/passivity tests, while the source \(B a\) can differ.
No measurement of \(C\), no response-Jacobian symmetrization, and no tighter
SCF residual can choose the correct \(a\).

With an energy-conjugate reciprocal continuum response \(Q\), the same
statement persists:

\[
\Omega_a(c;f)=
\frac12(c-a)^\mathsf T C^+(c-a)
+\frac12(Bc)^\mathsf TQ(Bc)+c^\mathsf Tf.
\]

On the declared support,

\[
\frac{\partial c_a^*}{\partial f}
=-\left[C^+ + B^\mathsf T Q B\right]^{-1}.
\]

That derivative remains independent of \(a\), whereas the on-shell energy,
surface potential, and continuum charge depend on \(B a\).  The executable
full-response KKT test locks this distinction: two reference potentials have
identical external response but different continuum charge and stationary
energy.

## 3. Consequences for existing V0 candidates

The raw MACE-POLAR fixed point is not a remedy.  Its field-conditioned
response failed the scalar/reciprocity gate, and the model is documented as a
non-self-consistent field formalism.  It cannot be renamed a stationary
electronic density merely because it emits a density-like tensor.

The frozen atomic-HF response table is also not a remedy by itself.  It is a
stronger induced-source representation than a guessed radial response and has
passed its registered one-acetone vacuum MEP source falsifier.  It remains
only a neutral tangent asset until an independently defined molecular
reference density and stationary scalar supply the affine origin.

Likewise, a free-atom promolecular density is a useful frozen source control,
not a molecular stationary state.  The already rejected QEq curvature and any
Harris-like nonselfconsistent construction cannot be repaired with a
hardness scale, response scale, or error-selected reference.

## 4. Constructive routes, without a hidden fit

There are only three honest classes of next step:

1. A frozen molecular field-energy model whose permanent density and response
   are both exact derivatives of one scalar.  The current MACE-POLAR
   checkpoint does not satisfy this condition.  MACE-Field illustrates the
   common-electric-enthalpy architecture, but its domain/training history must
   not be silently imported as a V0 molecular solution.
2. A separately specified stationary auxiliary electronic model that derives
   a molecular reference density, its nuclear pairing, and its response in
   the common density space.  It remains a V0 candidate only if it passes the
   physics and measured runtime gates; an on-the-fly QM calculation is not a
   production answer when it is slower than the declared QM reference.
3. A future trained variational electronic-functional head.  That is V1, not
   V0, and is explicitly deferred under the current no-post-training rule.

The distinction between an energy-functional model and a fixed-point model is
also the central architectural boundary identified by
[Baldwin et al.](https://arxiv.org/abs/2603.14700).  The original
[MACE-POLAR-1 paper](https://arxiv.org/abs/2602.19411) describes its present
formalism as non-self-consistent, while
[MACE-Field](https://arxiv.org/abs/2508.17870) is an example of deriving
electric response from a single learned electric enthalpy.  These references
motivate the gate; none supplies a permitted V0 permanent source by itself.

## 5. Admission rule

Before any physical continuum calculation, force/PES calculation, runtime
claim, or experimental solvation comparison, the candidate must expose:

* a stationary molecular scalar;
* a provenance-bound permanent electron reference in the same density/source
  dual space as the induced response;
* the exact nuclear source in that same total electrostatic ledger; and
* the existing joint KKT, reciprocity, passivity, coordinate-envelope, and
  force gates.

Until then the accepted use of the current full-response kernel is restricted
to a synthetic structural control.  It is deliberately not a total
solvation method and cannot be advanced by reducing a small benchmark error.

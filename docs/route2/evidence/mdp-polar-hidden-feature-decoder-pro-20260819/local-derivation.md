# Independent local derivation: hidden features do not identify a PCM source

This derivation was written independently while the verified Pro query was
running.

## Proposition

Let the hidden feature space be

\[
H=\bigoplus_l (\mathbb R^{m_l}\otimes V_l)
\]

and let a candidate atom-centred source space contain radial/multipole blocks

\[
Y=\bigoplus_l (\mathbb R^{n_l}\otimes V_l).
\]

Assume the released model fixes only a finite set of public observables
\(O:Y\to Z\), such as total charge, molecular dipole, and their uniform-field
response, while the continuum boundary map is \(B:Y\to\Sigma^*\). If there is
an equivariant map \(N:H\to\ker O\) such that \(BNh\ne0\) for an admitted
hidden state \(h\), then the family

\[
D_t=D_0+tN
\]

contains infinitely many distinct equivariant decoders with identical public
observables but different continuum boundary potentials. For any nondegenerate
quadratic continuum scalar on the image of \(BN\), at least two members also
have different continuum energies.

## Proof

Equivariance is preserved under addition and scalar multiplication, so every
\(D_t\) is equivariant. Since \(ON=0\),
\(OD_t=OD_0\) for every \(t\). Since \(BNh\ne0\), the boundary source varies
with \(t\). A nonconstant quadratic form restricted to that line cannot have
the same value for every \(t\). Thus symmetry and the finite public moments do
not identify the continuum source.

For the released MACE-POLAR architecture, an `l=2` hidden component provides a
concrete witness: adding an equivariant traceless-quadrupole decoder changes no
monopole or dipole. Making that perturbation independent of the uniform field
also leaves the public dipole polarizability unchanged. A generic cavity sees
its quadrupolar potential. The accompanying executable witness checks the
rotation covariance and the changed surface scalar numerically.

Even without hidden `l>=2`, two radial `l=0` densities can share the same total
charge and all exterior point-multipole moments while differing in the
penetration region sampled by a cavity. A finite moment set therefore leaves an
infinite-dimensional radial nullspace.

## Hidden-channel gauge

Within each repeated irrep block, the hidden coordinates may be changed by an
invertible channel transformation \(G_l\), with adjacent checkpoint weights
transformed by \(G_l^{-1}\), without changing the public input-output function.
A post-hoc decoder matrix \(A_l\) must be co-transformed as
\(A_lG_l^{-1}\) to preserve its value. Because no such decoder is included in
the released checkpoint contract, the public function does not select one.

Content-addressing one serialized checkpoint plus one hand-written decoder
makes a reproducible new profile. It does not turn that decoder into a
checkpoint-native physical observable.

## Why optimization principles do not cure identifiability

Minimum norm, Coulomb norm, maximum entropy, and least action select a section
only after a metric, prior, or functional has been supplied. Different
physically plausible choices give different sources in the nullspace of the
public observables. Unless that functional is itself part of the trained model
or follows from a proved physical representation theorem, the result is a
well-defined convention rather than an identified density.

## Narrow exceptions

A zero-training decoder would be defensible only if one of the following were
already checkpoint-bound:

1. trained frozen decoder weights with a declared physical basis and density or
   surface-MEP supervision;
2. an architecture theorem making a hidden block literally the coefficients of
   that physical basis, including normalization and radial meaning; or
3. sufficiently complete supervised boundary-potential/density data to remove
   the relevant nullspace.

The verified official contract exposes only `l<=1` q/p coefficients and does
not contain any of these exceptions.

## Local decision

Do not create a zero-training hidden-feature decoder. Finish the already frozen
canonical ADT experiment. If it fails its terminal gates, a density/MEP or
scalar-first response head must be a separately named model trained on
independent QM electrostatic data, not described as the unchanged checkpoint.

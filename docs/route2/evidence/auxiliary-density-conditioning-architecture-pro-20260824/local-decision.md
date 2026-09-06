# Local decision

Critically adopted:

1. The excellent ETB-2.0 reaction bound proves physical span, while the global
   condition failure proves nonidentifiability of the old molecular-fit
   coefficients. These are distinct claims.
2. For each element and l, freeze a symmetric isolated-atom radial transform
   `W_zl=K_zl^-1/2`, applied identically to every m. This is an SO(3)
   intertwiner, cavity-independent, and span-preserving.
3. Source/field duality transforms exactly: `c_old=W c_new` and
   `u_new=W.T u_old`, so `c_old.T u_old=c_new.T u_new`.
4. No geometry-dependent pivoting, eigenvalue deletion, SVD cutoff, or global
   molecular inverse may define model coordinates or training labels.
5. The same global constrained condition cap of 1e10 is retained after local
   whitening. If it still fails on any registered geometry, close this
   auxiliary-density route.
6. Exact charge is structural in the final scalar field chart. Dipole/first
   moment is supervised against QM response; the representation oracle may
   preserve it exactly, but a field-responsive model must not freeze it to MDP.

SALTED independently supports metric-weighted direct loss minimization over
nonorthogonal atom-centred bases, so global density-fit coefficient uniqueness
is not itself a physical-capacity requirement. The new architecture must train
on field energy/density/MEP observables rather than reuse the old global fitted
coefficient vectors as labels.

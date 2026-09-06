# Local decision

Critically adopted:

1. Candidate order is fixed by increasing total auxiliary dimension measured
   without any MEP or solvation result: ETB beta 2.0 (12196 total functions on
   the 12-case panel), AutoAux (13781), ETB beta 1.5 (20147).
2. The opened one-case AutoAux result does not alter that order.
3. Let `D=diag(J)`, `Jhat=D^-1/2 J D^-1/2`, and
   `A=C D^-1/2`. A full QR of `A.T` yields a nullspace basis `Z`. Numerical
   admission requires rank(A)=4 and cond2(Z.T Jhat Z)<=1e10.
4. The eigenvalue-ratio threshold is diagnostic/fail-closed, not a mode-
   truncation rule. The constrained fit is solved directly in the nullspace;
   no geometry-dependent pseudoinverse is used.
5. The same 12 cases may select the first development passer because no
   experimental solvation target is involved. Before results are opened, a
   fresh target-free confirmation identity must be frozen. Only the first
   development passer is evaluated once on confirmation.

The reaction-metric mean<=0.25 and max<=0.50 kcal/mol gates remain unchanged.
All three development candidates failing closes this standard-basis lane.

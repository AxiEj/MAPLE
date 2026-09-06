# Twelve-case auxiliary-density representation gate

This prospectively locked, no-solvation-target gate evaluated the standard
PySCF `df.make_auxbasis` Coulomb-fitting representation of twelve frozen
omegaB97M-V/def2-TZVPD densities. Electron count and electronic first moment
were constrained in a cavity-independent Coulomb metric.

The locked result is **FAIL**:

- mean weighted relative cavity-MEP error: 3.6692% (5% gate passed);
- maximum case error: 19.6319% for methane (10% gate failed);
- all other cases: 1.0501% to 3.7206%;
- maximum hard-constraint residual: 1.4566e-11 (1e-8 gate passed).

The methane absolute maximum error is 6.9436e-4 Hartree/e and the correlation
is 0.99698, so the failure is consistent with a low-reference-norm relative
metric rather than a broad auxiliary-basis collapse. That interpretation does
not alter the v1 result. A new gate, if any, must be prospectively frozen and
must use a physically allocated continuum-energy error budget rather than
changing this observed threshold.

No experimental solvation label was read, no model was trained or fit, and no
MAPLE capability is admitted by this evidence.

# Fixed local-whitened ETB-2.0 molecular condition gate

This was the single prospectively allowed follow-up after the standard
auxiliary-basis ladder failed. Fixed per-element/per-l isolated-atom Coulomb
whitening was generated and hashed before any molecular audit. No radial mode
was removed; the same transform was used for every m.

The result is **FAIL**:

- 10/12 molecules passed;
- benzene: global constrained condition 3.1813e10;
- aniline: global constrained condition 3.8526e10;
- fixed gate: every molecule <=1.0e10;
- maximum constraint-Gram condition: 41.93;
- maximum exact-constraint residual: 3.11e-14.

Local whitening improves the worst ETB-2.0 condition from 6.59e10 to 3.85e10
but does not close the preregistered gate. Per the frozen stop rule, the global
auxiliary-density coefficient route is closed: no cutoff, molecular pivot,
mode pruning, alternative whitening, or new standard auxiliary basis may be
tried as a rescue.

The prior reaction-metric evidence still proves excellent physical span. A new
architecture may use a smooth, analytic local partition and observable-level
QM supervision, but it must not reuse these globally fitted coefficients as
labels or claim this failed gate as passed.

No experimental solvation target or model output was read and no capability is
admitted.

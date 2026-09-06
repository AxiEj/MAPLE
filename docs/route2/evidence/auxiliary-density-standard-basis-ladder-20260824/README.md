# Standard auxiliary-density basis ladder

This target-free, prospectively locked ladder evaluated PySCF ETB beta 2.0,
AutoAux, and ETB beta 1.5 in cheapest-first order. All twelve development cases
were retained; a chemically disjoint confirmation identity was sealed before
any candidate result was opened.

The complete ladder is **FAIL**. Every candidate easily passed the PCMSolver
reaction-metric accuracy gate, exact moment constraints, and reciprocity, but
every candidate failed the diagonal-normalized, constraint-restricted Coulomb
condition-number cap of 1e10:

| candidate | mean bound | max bound | mean actual | max condition |
|---|---:|---:|---:|---:|
| ETB beta 2.0 | 0.032562 | 0.050952 | 0.006113 | 6.586e10 |
| AutoAux | 0.019140 | 0.036944 | 0.002322 | 1.911e11 |
| ETB beta 1.5 | 0.013880 | 0.024704 | 0.001209 | 3.391e13 |

All energies are kcal/mol. The frozen standard-basis ladder is closed: no
intermediate beta, changed cutoff, case deletion, or geometry-dependent
pseudoinverse is permitted under this experiment. The result nevertheless
shows that atom-centred auxiliary densities have ample physical MEP capacity;
the remaining issue is a stable, identifiable coordinate system for a future
scalar head.

No experimental solvation target was read, no model was fit or trained, and no
MAPLE capability is admitted.

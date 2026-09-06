You are a mathematical/physical-science auditor. We need a terminal theorem-level decision for a zero-training MACE-MDP + MACE-POLAR + ddPCM hybrid. Do not suggest fitting to solvation targets, arbitrary element/source scaling, a residual correction, or merely choosing a convenient decoder. Separate mathematical identifiability, representation/gauge invariance, physical meaning, and engineering provenance.

Official/checkpoint facts already verified:

1. The released MACE-POLAR-1 config has a local MACE backbone with `max_ell: 3`, but the trained physical electrostatic output has `atomic_multipoles_max_l: 1`, one 1.5-A Gaussian source width, `field_feature_max_l: 1`, field widths `[1.5, 3.0]`, and `quadrupole_feature_corrections: False`.
2. The official calculator exposes `density_coefficients` with shape `(n_atoms, 4)`: one monopole plus three dipole components. The official documentation explicitly warns that atomwise partial charges and dipoles are not unique.
3. MACE-MDP is officially a dipole/polarizability model, not an energy/force model. It does not expose a trained atomwise physical electron-density or cavity-MEP decoder.
4. The networks contain hidden equivariant atom features, including irreps up to l=3 in the MACE-POLAR backbone. These hidden tensors can technically be intercepted from the forward graph, but no released checkpoint head maps them to physical l>=2 multipoles, compact density, or cavity-surface electrostatic potential.
5. Existing target-independent tests show that q/p-only source modifications have reached a terminal accuracy/identifiability problem: the exact zero-field point-source prospective panel has fixed-source ddPCM MAE/q95/max = 1.46096/3.67088/4.53094 kcal/mol; a well-conditioned uniform-response manifold changes these only to 1.42833/3.54529/4.49363 and improves 32/60 cases. A learned MBIS prototype is not admitted and post-training is a last resort.
6. A separate canonical atomic-density-translation candidate is still running. We must not invent a second zero-training source merely because hidden features are available.

Formal setup:

Let a hidden atom feature space be

    H = direct_sum_l (R^{m_l} tensor V_l),

where V_l is the SO(3) irrep and m_l is its learned channel multiplicity. Let the desired continuum source/density space be

    Y = direct_sum_l (R^{n_l} tensor V_l),

including radial channels and possibly l>=2. A linear equivariant decoder has blocks A_l tensor I_{V_l}; nonlinear equivariant decoders are even less restricted. The public observables Q, molecular dipole mu, and polarizability alpha impose only a finite collection of moment/response constraints. Hidden multiplicity channels also admit internal basis changes G_l in GL(m_l) with compensating changes in adjacent network weights that leave the checkpoint input-output function unchanged.

Questions:

1. Prove or refute the following proposition: symmetry plus exact Q/mu/alpha closure cannot uniquely identify a basis/gauge-invariant decoder from hidden MACE features to a physical l>=2/radial density or cavity-MEP source. Give an explicit family of distinct equivariant decoders satisfying the same public observables but producing different cavity-surface potentials/PCM energies.
2. Explain the hidden-channel gauge issue precisely. Is a decoder defined from the numerical hidden coordinates of one serialized checkpoint at least content-addressed and reproducible, yet still not a checkpoint-native physical observable? Under what transformations would its meaning change while the checkpoint's public function remains invariant?
3. Does a minimum-norm, least-action, Coulomb-metric, maximum-entropy, or other canonical mathematical selection make the density physically identified without external density/MEP supervision? If not, distinguish a reproducible gauge choice from an identifiable physical quantity.
4. State all narrow exceptions. In particular, would zero-training use become defensible if the released checkpoint already contained a frozen trained density/MEP decoder, if an exact analytic theorem linked a hidden channel to a physical density, or if complete boundary-potential data were already among the supervised outputs? Are any such exceptions present in the facts above?
5. Does extracting hidden l>=2 features and attaching a new hand-designed decoder preserve the original MACE-MDP/MACE-POLAR model identity, or must it be named as a new source model/profile even if no parameter is fitted?
6. Give a terminal recommendation for the current hybrid accuracy program. Choose among:
   A. continue a zero-training hidden-feature decoder;
   B. close this loophole, finish the already-running ADT candidate, and if it fails move to a separately named scalar-first/density/MEP head trained on independent QM electrostatic data;
   C. another sharply defined zero-training exception, with the exact theorem that makes it identifiable.

We need a decisive answer, not general suggestions. State a theorem/counterexample, its assumptions, and any loophole. End with exactly one marker line:

HIDDEN FEATURES DO NOT IDENTIFY A PHYSICAL PCM SOURCE
or
A UNIQUE ZERO-TRAINING DECODER EXISTS
or
A NARROW IDENTIFIABLE EXCEPTION REMAINS

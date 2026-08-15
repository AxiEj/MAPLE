# Original MACE-POLAR source / frozen PCMSolver four-case audit

This directory retains two clean executions at Git head
`1d40c93bad67d6f4b4153f9bb8c7a981c391e3bf` of
`tools/route2_release/run_mace_original_source_pcmsolver_panel.py`.
Both executions reproduce scientific measurement SHA-256
`ff40c7c426cbcb59ab28378f29a1ed5675da98d4063fba20423b0247d997649b`.

The audit evaluates the official checkpoint's original four-channel source at
zero native field, uses its 1.5-A Gaussian monopole/dipole definition, and
compares its molecular electrostatic potential with the frozen total QM MEP on
exactly the same intrinsic PCMSolver IEFPCM cavity.  The fixed PCMSolver
response supplies the matched polarization energy.  No fitted charge model or
operational energy ledger enters this comparison.

## Result

All four records conserve total charge, but all four fail the inherited strict
`1 kcal/mol` fixed-source electrostatic budget that had been frozen before this
source test in the earlier representation-feasibility preregistration:

| compound | QM PCM (kcal/mol) | original-source PCM (kcal/mol) | absolute error (kcal/mol) | area-weighted surface-MEP relative error |
| --- | ---: | ---: | ---: | ---: |
| acetic acid | -10.983432 | -1.567258 | 9.416174 | 0.628145 |
| benzene | -3.136303 | -0.312064 | 2.824239 | 1.034156 |
| 2-acetoxyethyl acetate | -12.747357 | -0.919039 | 11.828319 | 0.754953 |
| acetone | -6.643339 | -0.866946 | 5.776393 | 0.670617 |

The mean absolute fixed-source error is `7.4612812780 kcal/mol`; the maximum is
`11.8283187109 kcal/mol`; `0/4` cases pass.  Molecular dipoles are much closer
(maximum relative error `0.0946640052`, with benzene close to the zero-dipole
limit), while the cavity-surface MEP and polarization magnitudes are grossly
underrepresented.  This is consistent with the learned residual source being
useful for MACE's own long-range decomposition but not, unchanged, a full
quantitative PCM source.

## Decision boundary

This closes the unchanged original-four-channel source for the current
quantitative separated PCM profile.  Therefore neither `Phi0` nor `Phi1Delta`
may proceed to ledger, force, PES, or public admission on this source identity.
It does not claim that MACE-POLAR's internal decomposition is wrong.  A
separately named fixed radial embedding remains only a possible research
branch; it is not authorized until independent quadrupole and far-field
point-charge gates show that the missing physics is radial-only.  No
`E/F/H/V/M` capability is admitted.

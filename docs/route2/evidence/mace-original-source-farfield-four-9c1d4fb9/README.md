# Original MACE-POLAR source far-field four-case audit

This directory retains two clean executions at Git head
`9c1d4fb961a7d1b06372d6dcd86c165e582394c9` of the preregistered
`tools/route2_release/run_mace_original_source_farfield_panel.py`. Both runs
reproduce scientific measurement SHA-256
`dc9cce0f1ced9c34af1ab67bdae329d4521f2e67b780a1adcad6b7dd645d97d8`.

The reference is evaluated directly from the frozen
omegaB97M-V/def2-TZVPD checkpoint density. No new SCF, solvent, PCM energy,
experimental solvation label, fitted charge, or energy-ledger result enters the
audit. Multipoles use the nuclear-charge centroid. Three deterministic
far-field shells lie 8, 12, and 16 A beyond the maximum nuclear extent.
Thresholds and the four-record panel were committed before execution in
`docs/route2/preregistrations/mace-original-source-farfield-four-v1.json`.

## Result

All four cases pass the preregistered necessary-condition gate:

| case | dipole relative L2 | quadrupole relative Frobenius | maximum shell relative L2 |
| --- | ---: | ---: | ---: |
| acetic acid | 0.0155871 | 0.3584262 | 0.0239048 |
| benzene | 0.0946640 | 0.1426431 | 0.1426207 |
| 2-acetoxyethyl acetate | 0.0126363 | 0.0413566 | 0.0381044 |
| acetone | 0.0302635 | 0.0589589 | 0.0345747 |

The acetic-acid quadrupole relative error is large, but its absolute Frobenius
error (`0.39436 e A^2`) remains below the threshold frozen before execution
(`0.25 e A^2 + 0.20 * ||Q_ref|| = 0.47005 e A^2`). The benzene far-field shell
is the tightest case and remains below the preregistered combined absolute and
relative error limit on every shell.

## Decision boundary

This result does **not** reopen the unchanged four-channel source: that source
still fails the matched QM/PCMSolver cavity-near-field gate by 2.824--11.828
kcal/mol. It establishes only the necessary low-multipole/far-field condition
for exactly one separately named, fixed, symmetry-preserving radial-embedding
research experiment. It does not prove that the near-field defect is
radial-only; held-out cavity-surface MEP and fixed-source PCM energy remain
mandatory. `Phi0`, `Phi1Delta`, and all E/F/H/V/M capabilities remain disabled.

# Zero-training Ewald gauge-separated permanent source: opened-tail result

Date: 2026-08-17

## Frozen candidate

The parameter-free hybrid source was locked before evaluation as

```text
V = V_point[c_POLAR(0)]
  + V_Gaussian,sigma=1.5A[c_MDP - c_POLAR(0)].
```

The only radial scale is the official MACE-POLAR charge-density width.  No
checkpoint was changed, no source/solvent/CDS parameter was fitted, and no
experimental solvation target was read.  The seven geometries were the already
opened largest errors of the earlier 60-case source gate, so this is a
falsification diagnostic and not an independent accuracy estimate.

## Result

The construction exactly closes the MACE-MDP far-field charge and molecular
dipole, and improves the aggregate q/p/Q/O cavity-MEP relative RMSE from
`0.28813598` to `0.27468630`.  It nevertheless worsens the q/p-only fixed-source
ddPCM energy MAE from `3.71305394` to `4.48888138 kcal/mol`, improves only
`2/7` cases, and raises the maximum error from `4.53093611` to
`7.81187236 kcal/mol`.  It therefore fails its prelocked gates and is not
integrated or tuned.

This terminal result is specific to compatibility with the q/p-only energy
reference.  Since the candidate was designed to change near-field information,
a distinct q/p/Q/O ddPCM oracle is required to answer whether its improved
higher-multipole MEP corresponds to improved continuum energy.  That is a new,
prospectively locked scientific question; it does not alter this rejection.

## Claim boundary

- no energy, force, solvation, or public capability is admitted;
- no pure MACE-POLAR claim is made;
- SPICE MBIS q/p/Q/O is evaluation-only;
- no MNSol, FreeSolv, CDS, standard-state, or confirmation datum is used.

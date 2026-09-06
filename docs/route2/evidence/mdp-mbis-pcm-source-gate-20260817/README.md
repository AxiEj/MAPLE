# MDP-MBIS PCM permanent-source gate

This is a 60-molecule-held-out, target-free source diagnostic. It compares the
original latent MACE-MDP atomwise q/p partition and a frozen-backbone MBIS
source head against independent SPICE MBIS multipoles on identical ddPCM cavity
nodes. It is not a solvation-accuracy, direct full-density-QM-MEP, force, or
public-capability result.

The prospective gate failed. The learned head reduced fixed-source ddPCM energy
MAE from 8.191931 to 2.534650 kcal/mol and improved 58/60 configurations, but
only reduced the primary radius-1.0 q/p MEP global RMSE by 7.43% (required 50%)
and worsened the MEP at larger radius scales and against q/p/Q/O.

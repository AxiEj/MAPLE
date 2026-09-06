# MDP-polarizability dipole-closure tail falsification

This directory freezes an adversarial, explicitly non-independent seven-case
test of one parameter-free MACE-MDP + MACE-POLAR source composition.  The cases
were the seven largest previously opened fixed-source errors of the rejected
uniform-response candidate.  The new candidate itself had not been evaluated
when its formula, inputs, and terminal gates were locked.

The candidate keeps the MACE-POLAR zero-field atomic charges and applies the
unique uniform linear-response dipole correction supplied by the frozen
MACE-MDP polarizability decomposition:

```text
E_equiv   = alpha_total^-1 (mu_MDP - mu_POLAR)
delta p_A = alpha_A E_equiv
```

It has no fitted parameter, reads no experimental solvation target, preserves
total charge, closes the MACE-MDP molecular dipole, and is rotation covariant.
Individual atomwise polarizability contributions are not reinterpreted as
standalone positive atomic tensors; only their exact additive identity is used.

Frozen decision:

```text
candidate-rejected-by-opened-tail-falsification
```

Results:

```text
                                  POLAR zero    candidate
tail MAE / kcal mol-1               3.7130539     3.6379521
tail maximum / kcal mol-1           4.5309361     4.5017350
global q/p/Q/O relative MEP RMSE    0.2881360     0.2884257
paired improvements                                5 / 7
relative MAE reduction                             2.02%
```

The candidate fails the prelocked 10% material-improvement gate and worsens
the cavity MEP.  It is therefore rejected without a new independent panel and
must not be rescued by a fitted scale, eigenvalue clipping, or source mixing.

File SHA256 of `aggregate.json`:

```text
95b8704a59360d1508e4338544b899068347c5e8efd16e71c1e2ae4e0fb2b6af
```

Canonical payload `aggregate_sha256`:

```text
de72c4e1fa876b3e7838149b886993921bb78dbb9ea2873c3c67c5c858ae13b4
```

Preregistration SHA256:

```text
1a50b56029216c2b1ed298ce9c759e2289a00b1b054079e02d06ed4bd0c93d96
```

No public E/F/H/V/M capability and no permission to post-train follows from
this evidence.

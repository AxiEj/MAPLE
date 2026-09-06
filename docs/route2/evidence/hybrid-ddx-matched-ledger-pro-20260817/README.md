# Hybrid matched QM/ddPCM ledger — Chrome Pro checkpoint

This directory freezes the third focused mathematical-advisor checkpoint for
the MACE-MDP + MACE-POLAR hybrid ddPCM route.  It was submitted through the
retained Windows Chrome session only after the UI visibly passed all genuine
Pro checks:

```text
composer=Pro
power=Pro, 5 of 5.
new_chat=True
generation_observed=True
```

The complete prompt, submission record, and unabridged response are retained;
the answer ends with the requested marker
`HYBRID MATCHED QM-DDPCM LEDGER DECISION`.

## Content hashes

```text
q17-prompt.txt      4b6c89a4c7eac0bab64746b942463be268b64b3f40155623bfb33862121724fb
q17-submission.txt  fef7add3eefaaaa21f78fbe97d91d9616093daa35ffa643d7726ed8ee087f968
q17-answer.md       90c5fa32b1ee625720d10d7792179db939f128bcde7c46814f9232af813ded73
```

Conversation:
`https://chatgpt.com/c/6a826c35-c788-83e8-bdd4-c8f157363b99`

## Accepted narrow decision

The earlier four-molecule QM/PCMSolver IEFPCM/GEPOL decomposition is a useful
cross-equation sensitivity result, but it cannot select between `Phi0` and
`Phi_raw`.  The required terminal comparison is a target-independent,
same-geometry and same-discrete-operator QM/pyddx-ddPCM decomposition:

```text
D_QM = E0[gamma_pcm] - E0[gamma_vac]
S_QM = G_dd[gamma_pcm]
T_QM = D_QM + S_QM
```

Only solvation shifts are compared:

```text
DeltaPhi0   = G_cont_ML
DeltaPhiRaw = DeltaE_conditioned_raw + G_cont_ML
```

The raw field-conditioned energy difference is not identified with `D_QM`.
Passing the matched total ledger may retain one operational scalar and its
implicit conservative force; it cannot reopen strict Tier V.

## Immediate experiment

Run the frozen benzene, acetone, acetic-acid, and diester geometries with the
same water radii, `lmax=15`, `nleb=1202`, ddPCM equation, metric, and sign
convention on both QM and hybrid sides.  Record the direct QM boundary source,
the permanent/response/feedback source-error split, both operational ledger
residuals, stationarity, dual-work identities, two target-independent PCM-SCF
starts, and a cold replay.  No experimental hydration target, CDS, standard
state, conformer weight, or nonpolar term may be read.

This Pro answer is an external mathematical review.  Its decision is not an
admission result until the matched calculation is implemented and independently
verified locally.

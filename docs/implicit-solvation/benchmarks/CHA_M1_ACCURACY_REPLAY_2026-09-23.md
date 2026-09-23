# CHA M1 accuracy replay — 2026-09-23

## Question and endpoint boundary

Does commit `c744b140` change the fixed-geometry water-solvation predictions
previously reported for AM1-BCC + CHA-GB/GBNSR6 + PBSA `inp=2`
(`EGB + ECAVITY + EDISPER`)? **No.** M1 adds a separate Torch *polar-algebra*
module, but does not register it as a solvent provider or change the existing
AmberTools runtime. The tests below independently reran the historical scalar
endpoint and checked the new algebra against full-precision native EGB. They do
not test complete Torch forces, an analytic Hessian, or a new solvation model.
The existing `amber_chagb.py` provider and `run_chagb_nonpolar.py` benchmark
runner have identical Git blob IDs at baseline `2839ddd0` and `c744b140`.

All energies below are kcal/mol. The same pinned FreeSolv GAFF geometry, AM1-BCC
charge vector, GAFF2/Bondi topology, zero-salt CHA profile, and PBSA nonpolar
profile were retained. No new QM, MLIP training, experimental fit, or model
selection was performed. The development and historically label-exposed reserve
sets are **not new blind confirmation**.

## Fixed ten-class pilot

The ten chemical classes and IDs came from the pre-existing
[`freesolv10-2026-07-22.json`](freesolv10-2026-07-22.json)
(SHA256 `619ea1eb4e8a86d0cbc880012d86a5d233bf38053828bdc7f57d7ad2c89117b6`),
not from current errors. All ten source MOL2 hashes and saved AM1-BCC charge
vectors matched the frozen 526-case source manifest; the pinned AmberTools
executables were verified before evaluation.

| Class | Molecule | Experiment | Old CHA/PBSA | Current CHA/PBSA | Absolute error |
| --- | --- | ---: | ---: | ---: | ---: |
| Alkane | Methane | 2.00 | 2.1627 | 2.1627 | 0.1627 |
| Aromatic hydrocarbon | Benzene | -0.90 | -1.5806 | -1.5806 | 0.6806 |
| Alcohol | Methanol | -5.10 | -3.6394 | -3.6394 | 1.4606 |
| Ether | Dimethyl ether | -1.91 | -1.6732 | -1.6732 | 0.2368 |
| Ketone | Acetone | -3.80 | -5.1324 | -5.1324 | 1.3324 |
| Ester | Ethyl acetate | -2.94 | -5.1770 | -5.1770 | 2.2370 |
| Nitrile | Acetonitrile | -3.88 | -3.1895 | -3.1895 | 0.6905 |
| Aromatic amine | Aniline | -5.49 | -7.1712 | -7.1712 | 1.6812 |
| Haloalkane | Chloroethane | -0.63 | 0.0029 | 0.0029 | 0.6329 |
| Sulfoxide | Dimethyl sulfoxide | -9.28 | -8.9177 | -8.9177 | 0.3623 |

Ten of ten scalar predictions were exactly equal to their historical records:
MAE **0.9477 → 0.9477**, RMSE **1.15298 → 1.15298**, maximum absolute error
**2.2370 → 2.2370**. This tiny pilot is not a basis for claiming CHA is more
accurate than OBC: the historical OBC-II/ACE MAE on these same ten was 0.6797.

For each of these ten molecules, the verified, trace-instrumented native
GBNSR6 exposed the *pre-shift inverse Born radii* and effective CHA radii.
Feeding exactly those inputs into M1 Torch yielded a maximum absolute
Torch-versus-native EGB difference of **2.04×10⁻¹⁴**. The maximum difference
between full-precision native EGB and the historical four-decimal printed EGB
was **4.77×10⁻⁵**. Replacing only that rounded polar number while leaving the
old rounded PBSA terms fixed gives a *diagnostic arithmetic splice* with MAE
0.94769094 (a change of −9.06×10⁻⁶); it is **not** a new matched endpoint or
an experimentally meaningful accuracy gain.

## Complete development and reserve replays

| Frozen panel | Old MAE | Current MAE | Old RMSE | Current RMSE | Numerical identity |
| --- | ---: | ---: | ---: | ---: | --- |
| FreeSolv development, 526 | 1.321851711 | 1.321851711 | 1.854207939 | 1.854207939 | 526/526 result JSON files byte-identical; summary SHA256 identical |
| Label-exposed reserve, 116 | 1.300816379 | 1.300816379 | 1.845626715 | 1.845626715 | Maximum component/total delta ≤3.56×10⁻¹⁵ from unit round-trip |

The 526-case rerun used the existing `run_chagb_nonpolar.py` energy and summary
paths with `--workers 1`, a fresh output directory, and the original frozen
charge manifest. All 526 completed. Its entire summary file was byte-identical
to the historical summary (SHA256
`a842604d423dd01833b7e4a2b2d92ce0944e0d0e6f2c2e1d883edeaf2cf7308a`).
The old/current maximum absolute error was 9.4164.

The standard reserve runner would **reinvoke Antechamber/SQM** in a fresh
directory, contrary to the no-new-QM boundary. Instead, the 116-case replay
read each previously sealed AM1-BCC charge vector, verified its hash and the
pinned source MOL2, and called the current `AmberToolsChaGB` provider with
that fixed vector. The old sealed-v1 energy artifact (SHA256
`5439b65f047d0aafa14282dfa47df8f0a05d5d965cce9cc3a52caf5b5eec3c09`)
was the comparison baseline. All 116 ultimately completed; the maximum polar,
cavity, dispersion, and total discrepancies were respectively
1.78×10⁻¹⁵, 3.56×10⁻¹⁵, 3.56×10⁻¹⁵, and 1.78×10⁻¹⁵. The maximum absolute
error remained 6.3763. This reserve's labels had already been exposed, so its
unchanged score is a replay, **not** fresh generalization proof.

One reserve molecule, `mobley_8117218`, hit a native GBNSR6 double-free on
its first attempt. A retry from the same frozen source MOL2, saved charge
vector, and provider configuration in a fresh audit directory succeeded.
The attempt statuses and failure text are retained in the energy report, but
the failed attempt's prepared native input files were not retained; bytewise
identity between the two prepared input bundles is therefore **not** proven.
This is a native reliability warning, not an accuracy improvement.

## Provenance and retained evidence

The development protocol/source-manifest SHA256 values are
`524cbcb66e9ee59ee502c59c59a186a35607f6cde0ac1ed43007a01898f6c9f8` /
`84cd6569296ab4f0ae8410f3f03728fd04c358980936b811be57e8ff2c2508dc`.
The reserve protocol/source-manifest SHA256 values are
`9cf657e256752b5b00b7b7b631efdbc54d9a2c7134da80e8eb97bd96fa70b0c5` /
`24969fefd4e41b18f6f7c7c85523d2ec9bc888ecee882c75783e8526a95f1c81`.
Executable SHA256 identities are frozen in those protocols. No historical
canonical artifact was overwritten.

Detailed records, the ten-case Torch/native trace, reserve attempt log, and
the bounded replay scripts are retained locally under
`.omx/benchmarks/route1-cha-m1-accuracy-replay-20260923/` (ignored by Git).
The key report SHA256 values are:

- `panel10/score_report.json`: `62ff5f329f28868714b42e4ee70fcbec656ac3470fa945340196a0f77287b184`
- `panel10/torch_polar_trace_report.json`: `d8f3494283dcc4086f1c79eab25569b7e8a756273c48699cb10ec3c9ff4c06c1`
- `development526/summary.json`: `a842604d423dd01833b7e4a2b2d92ce0944e0d0e6f2c2e1d883edeaf2cf7308a`
- `reserve116/energy_replay.json`: `13d02801076a8b3bf9e77e76b0d8593fcd12ebb62812dbbda202ad4502c08434`
- `reserve116/score_report.json`: `8ef8936b3089b9bf3dea2ee93baf67b1ee91ee147f2622f0a78202e2cd6f6516`

**Conclusion:** the CHA/PBSA scores on these frozen panels have not changed.
M1 reproduced
the polar algebra on a chemically diverse ten-case panel, but remains
unregistered and cannot establish complete coordinate forces or a new
experimental MAE. The contrast between the ten-class pilot and all 526
development cases is another reason not to select a model by a small panel.

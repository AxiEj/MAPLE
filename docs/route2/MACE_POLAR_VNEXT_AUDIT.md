# MACE-POLAR conservative-vNext audit

## Result

The vNext contracts, reduced state equation, implicit adjoint, fixed-topology
C-PCM backend, and official MACE-POLAR-1-M adapter are implemented and tested.
No Route-2 capability is admitted. The current official checkpoint does not
expose the one exact checkpoint-native conjugate source/receiver operator
required by the production profile.

The unique production-target scalar remains

\[
E_{\rm op}(R)=E_{\rm vac}(R)
+\tfrac12\langle c^*(R),P_R(c^*(R))\rangle_Q,
\]

where \(c^*=c_{\rm ref}+Ty^*\) and \(y^*\) is the unique admitted root of

\[
T^+\left[c_{\rm ref}+Ty-
\Pi_qM_\theta\!\left(R,P_R(c_{\rm ref}+Ty)\right)\right]=0.
\]

The field-conditioned MACE energy difference and nonpolar/CDS terms are not
part of this scalar. The total derivative, if eventually admitted, is defined
only through the matching implicit adjoint. There is no second force formula.

## Exact-GTO gate outcome

The official checkpoint exposes:

- learned source: one radial width, `sigma=(1.5 Angstrom)`, \(l\leq1\),
  `normalize=multipoles`, four values per atom;
- learned receiver: two radial widths, `sigma=(1.5, 3.0 Angstrom)`,
  \(l\leq1\), `normalize=receiver`, eight features per atom;
- upstream receiver matrix shape: `(8, 4)`.

Those spaces cannot be treated as one \(B/B^*\) pair merely because a matrix
connects their shapes. The current adapter therefore declares
`exact_gto_operational_available=false`. It does not symmetrize, fit, or invent
a missing map. The separately named local-jet path remains diagnostic only.

## Executed real-checkpoint canary

The committed evidence bundle tests clean implementation commit
`19eea1ca99028f8b432e6f08a5c64d87a58f4011` and is stored under
`docs/route2/evidence/vnext-mace-polar-gate-19eea1ca/`.

Environment: Python 3.11, Torch 2.12.0+cu130, CUDA, `mace-torch==0.3.16`,
`graph-longrange==0.4.0`, float64, official `polar-1-m` checkpoint SHA256
`fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`.

Command:

```bash
export PYTHONPATH=/home/axie/.cache/uv/archive-v0/F-d7m2HPEUDaGHp7LnIsM:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
export MAPLE_ROUTE2_REAL_MACEPOL=1
export MAPLE_ROUTE2_MACE_DEVICE=cuda
python -m pytest -q -s --disable-warnings \
  tests/route2_vnext/test_mace_polar_real_checkpoint.py
```

Observed:

| check | value |
| --- | ---: |
| uniform local-field source parity | max absolute error `0.0` |
| nonuniform-field source change | L2 `1.5563605803939204e-4` |
| source JVP/VJP dot error | `2.554920623752243e-13` |
| source-position VJP directional error | `2.4787356743549704e-7` |
| total source charge | `2.7755575615628914e-17 e` |
| local versus upstream field-conditioned energy branch | `-0.0038528325721927104 eV` |

The last value is deliberate negative evidence: the field-conditioned model
energy is not silently added to the operational half-coupling scalar.

## Methane fixed-cavity diagnostic

A disabled local-jet scalar/profile was added solely so the new kernel can be
compared with the historical fixed-cavity implementation without borrowing the
exact-GTO production identity.

For FreeSolv `mobley_9055303` (methane), water dielectric 78.39, amplitude-SWIG
Lebedev order 15, and the official float64 checkpoint, the vNext diagnostic
converged in seven updates:

| measurement | result |
| --- | ---: |
| physical reduced residual norm | `7.96785980794461e-12` |
| total charge | `0.0 e` |
| C-PCM half-coupling | `-1.483425546707858 kcal/mol` |
| historical same local-jet L15 result | `-1.4834255431954178 kcal/mol` |
| difference | `-3.5124403385822234e-9 kcal/mol` |
| half-coupling identity error | `1.3877787807814457e-17 eV` |
| cold replay field max difference | `0.0 eV` |

This is numerical migration parity, not an accuracy improvement. The old
atom-centred SMD-CDS contribution is not included, so this is not a complete
\(\Delta G_{\rm solv}\). For context only, the earlier rho-DROP methane
electrostatic result was `-0.23949099688080341 kcal/mol`; it uses a different,
source-dependent cavity and also lacks an admitted nonpolar term, so the two
numbers are not interchangeable model-quality scores.

## Capability decision

| capability | status | reason |
| --- | --- | --- |
| scalar energy E | closed | no exact checkpoint-native production coupling/admission artifact |
| conservative force F | closed | production scalar cannot be assembled under the required coupling identity |
| Hessian/FREQ H | closed | depends on admitted F |
| strict variational V | closed | energy-source conjugacy/stability not established |
| MD M | closed | depends on admitted F plus path/NVE gates |
| OPT/NEB/TS/IRC | closed | no Tier-F profile |

The next scientific step is to obtain or train a model whose response source
and receiver are one declared energy-conjugate operator, or to establish a new
formally justified source space. Solver tuning cannot repair this mismatch.

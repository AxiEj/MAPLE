# Real-checkpoint MACE-POLAR conjugacy no-go evidence

## Decision

At clean execution commit
`d17c35acb3de71e9780741d57b476de3f589d9d7`, the official
MACE-POLAR-1-M checkpoint produced a material energy-gradient component in the
field subspace that is orthogonal to the embedded original four-channel source.
This is a counterexample to the field-energy/source identity for both possible
signs.

Therefore the following global claim is formally ruled out:

> retain the original checkpoint intrinsic field-conditioned energy and retain
> the original four-channel density/source head unchanged while treating both
> as derivatives of one scalar on the full two-width eight-channel field space.

All E/F/H/V/M capabilities remain false. This result does not admit the new
energy-gradient effective-source model; that is a different model identity and
still requires its own stationarity, gauge, passivity, force, and release gates.

## Executed command

```bash
python tools/route2_release/run_mace_conjugacy_nogo.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --output /tmp/route2-mace-conjugacy-nogo-d17c35ac-run2.json
```

The identical command was repeated to a second output path as a cold replay.

## Bound inputs

| item | value |
| --- | --- |
| execution Git HEAD | `d17c35acb3de71e9780741d57b476de3f589d9d7` |
| working tree | clean |
| checkpoint SHA256 | `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a` |
| device | CUDA |
| inference dtype | float64 after explicit checkpoint conversion |
| geometry | fixed neutral singlet water canary |
| states | zero field; deterministic nonzero eight-channel field |
| measurement SHA256 | `d89b620ed31cfdc0dfd7a89d03551251008458fe62e60c4eacda020255ab6909` |

The JSON artifacts contain the exact source hashes, checkpoint record, runtime,
field transform, source embedding, spaces/pairing, geometry, protocol, raw
vectors/matrices, and decision record.

## Decisive measurements

| measurement | zero field | nonzero deterministic field |
| --- | ---: | ---: |
| energy-gradient L2 | `0.4232215784` | `0.4230873918` |
| missing-radial witness L2 | `0.3143613123` | `0.3142884952` |
| missing-radial witness relative | `0.7427818625` | `0.7428453347` |
| gauge-reduced missing witness L2 | `0.3096925107` | `0.3096212958` |
| gauge-reduced missing witness relative | `0.7317502851` | `0.7318140455` |
| witness threshold | `5.2322e-10` | `5.2309e-10` |
| susceptibility reciprocity relative defect | `1.2573947688` | `1.2569246697` |
| original-source constant-potential response L2 | `3.8819e-3` | `3.8821e-3` |

For direct conjugacy, both `s=+1` and `s=-1` fail in the full and
gauge-reduced spaces. Relative residuals range from `1.376` to `1.411`.
For the intrinsic-energy stationarity identity, both signs fail with relative
residuals approximately `1.0`.

The original preregistered central finite-difference gate fails at both states
because differences of the roughly 2-keV total energy are cancellation-limited.
That failure is retained as negative evidence and is not relabelled. The added
forward/reverse AD implementation cross-check agrees:

- zero field absolute AD difference: `1.4236033e-10 eV`;
- nonzero field absolute AD difference: `6.4482517e-12 eV`.

Thus the no-go witness is not accepted on an unverified reverse-mode gradient.

## Cold replay

The two runs have different wall times and therefore different whole-file
hashes, but their complete `protocol`, `states`, `decision`, and
`measurement_sha256` objects are exactly equal.

| artifact | SHA256 |
| --- | --- |
| `measurements.json` | `3fcc90a01f6fb76d1ddf9a60ad296c8d07602baea9ed24c76a56a5af60b77bb6` |
| `cold-replay.json` | `95d46a0d09427aadf53666448d74dd414739113beda19c3a68ece81082b7cf47` |
| `runner.log` | `63f3e8904fb2c3990ccf9f68f51442c017ca45ef7385e4877e2b7baf7e467b20` |
| `cold-replay.log` | `185226694e37b0b25a73a5a066c388e450d6d68afb97791c929db45f0d7ff893` |

The first run took `12.01 s` inside the runner (`13.13 s` wall time) and used a
maximum resident set of `2,100,764 kB`. The replay took `10.48 s` inside the
runner.

## Claim boundary

One counterexample is sufficient to disprove the claimed global common-scalar
identity. It does not prove anything about chemical accuracy, total solvation
free energy, conservative nuclear forces, root uniqueness, or the separately
constructed eight-channel energy-gradient effective source.

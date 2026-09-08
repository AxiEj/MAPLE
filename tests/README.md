# Scientific regression tests

Use `pytest` in the same environment as the checkout (the test runner must be installed separately). The lightweight numerical
tests use MAPLE's declared NumPy/ASE/SciPy dependencies.
Tests needing optional runtimes call `pytest.importorskip`; tests needing model
weights additionally require explicit local paths. No test downloads or
re-exports a model.

```bash
python -m pytest -q
```

## Native backend parity

These tests compare MAPLE with the upstream runtime using the *same weights*.
They skip unless their assets are provided:

```bash
MAPLE_TEST_MACE_NATIVE_MODEL=/trusted/MACE-POLAR-1-S.model \
MAPLE_TEST_MACE_TRACE_MODEL=/trusted/macepols.pt \
MAPLE_TEST_AIMNET2_MODEL=/trusted/aimnet2.pt \
MAPLE_TEST_AIMNET2NSE_MODEL=/trusted/aimnet2nse.pt \
MAPLE_TEST_ANI_MODEL=/trusted/ani1x.pt \
python -m pytest -q -m backend tests/test_backend_parity.py
```

The MACE test first proves that every eager-model state tensor occurs unchanged
inside the traced wrapper, then compares energy, forces, and monopole charges
for singlet, doublet, and triplet inputs. Only the S-size checkpoint is covered;
this is not evidence for separately trained M/L weights. The AIMNet2 tests use
the official `aimnet` runtime and compare both AIMNet2 and AIMNet2-NSE in their
full-pair Coulomb and DSF modes with the MAPLE wrapper. These are optional
integration tests: a skip is not parity evidence.

The native comparisons were exercised with MACE 0.3.16 and AIMNet 0.2.0.
Other runtime versions must pass the same weight-identity and numerical assertions.

The ANI test uses real ANI and D4 evaluations to prove that changing the D4
setting invalidates a previously cached energy.

Only point the MACE native-model variable at a trusted local file because the
upstream loader must deserialize its Python model object.

## Statistical check

```bash
python -m pytest -q --run-slow tests/test_npt_statistics.py
```

This checks the mean and variance of a controlled ideal-gas C-rescale volume
distribution at two timesteps. It is a reproducible algorithm test, not
validation of a particular material, calculator stress, or production NPT
setup.

Generated output, logs, and model files do not belong in `tests/` or commits.

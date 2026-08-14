# Disabled real-checkpoint common-scalar water canary

This directory preserves two clean-checkout executions of the changed-source
MACE-POLAR common-stationarity candidate at commit
`576550e9cfe1011519c7bd93e7ee256972ff2560`.

## Exact command

```bash
python tools/route2_release/run_variational_common_water_canary.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --output /tmp/route2-variational-common-water-576550e9-run1.json
```

The second execution changed only the output suffix to `run2`.

## What passed

- Official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`.
- The disabled common state converged in 32 Anderson iterations to an actual
  unmixed reduced residual of `1.104397210579806e-10`.
- Re-evaluation from that root was accepted immediately as a warm root; source
  and stationary energy were identical under the registered replay contract.
- The three central differences at `5e-4`, `2e-4`, and `1e-4` Angstrom all
  matched the stationary-envelope directional derivative.  The best absolute
  error was `4.157446151231703e-09 eV/Angstrom` at `2e-4 Angstrom`; the largest
  was `2.9895738047208686e-06 eV/Angstrom`.
- Both independent executions produced the exact same measurement SHA-256:
  `62d43c63868270fc74254cf0ddbc182b33af0d158c3fbca1ef6493cdf52ee9fc`.

The stationary ledger at the central root was:

```text
model energy       -2079.8973328901448 eV
coupling              -0.03357459874936536 eV
continuum energy      -0.016787299374682672 eV
stationary total   -2079.88054559077 eV
```

The equality `continuum energy = 0.5 * coupling` holds for this fixed linear
reciprocal continuum.  With conjugacy sign `s=+1`, the recorded total is
`E_model - coupling + G_continuum`.

## What this does not prove

This is one water geometry and one coordinate direction.  It uses a changed,
complete eight-channel field-energy-gradient effective source; the original
four-channel density head is only the zero-field anchor and a diagnostic
observable.  It does **not** rehabilitate the formally ruled-out original
energy/original-source pairing.

The continuum is the fixed-topology amplitude-SWIG implementation assembled on
one laboratory-fixed 194-point Lebedev rule.  Conventional finite point
sampling is not a structural SO(3) representation.  Consequently this result
does not prove global rotation covariance, passivity, root uniqueness,
combined-Hessian stability, moving-cavity derivatives, chemical accuracy, or a
production PES.

Every `E/F/H/V/M` capability remains false.  These files are implementation
and replay evidence only, not an admission artifact.

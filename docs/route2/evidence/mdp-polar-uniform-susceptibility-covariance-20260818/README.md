# Target-free MDP/POLAR uniform-susceptibility covariance audit

This is an exploratory structural falsification test, not an accuracy or
capability admission.  It reads only atomic numbers and coordinates for the
already-opened `08P ASN` geometry.  It does not load MBIS, energy, force, or
solvation targets and performs no fitting or post-training.

The candidate keeps the MACE-POLAR atomwise source topology and uses the
public MACE-MDP **molecular** polarizability only to reparameterize POLAR's
three-dimensional affine-potential response.  The native field chart reads
only the duplicated gradient channels; potential values and gradients are not
mixed in an Euclidean norm.

## Command

```bash
python tools/route2_release/audit_mdp_polar_uniform_susceptibility_covariance.py \
  --device cuda \
  --output result.json
```

The audit independently reevaluates both checkpoints after one fixed proper
rotation, translation, and atom permutation.

## Result

| Metric | Relative error |
| --- | ---: |
| MDP molecular polarizability covariance | `3.83e-16` |
| POLAR zero-field source covariance | `9.05e-09` |
| POLAR uniform source-Jacobian covariance | `9.97e-09` |
| derived 3x3 coordinate-transform covariance | `8.01e-10` |
| corrected native-field covariance | `1.95e-10` |

The transform condition number is `1.2957146394` before and after the rigid
operation.  Thus this geometry provides no covariance counterexample; the
observed floor is set by the real MACE-POLAR source/JVP evaluation, not the
three-dimensional algebra.  This says nothing about chemical accuracy,
passivity, self-consistent-root stability, an energy ledger, or nuclear forces.

## Content binding

```text
result.json
  SHA256 7a1b9b56e40ac05ca3bacf263d5902c266287cf40b2e4903e60826d1a61589a1

audit runner
  SHA256 f68f9720364ae796dca2fc692d51b81c87461a4d94f8c1255f92c2213b9f8174

susceptibility implementation
  SHA256 24eb0a8d01be387baff676f2faed5cb34f56b618b838d167ba8b2318eb95817a
```

`result.json` additionally binds both checkpoint files, the SPICE container,
the frozen selection artifact, both model adapters, the runtime, the exact
geometry identity, and its own canonical record digest.

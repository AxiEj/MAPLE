# LSNN upstream parity record

This note records the **reviewed upstream parity evidence** for the
`lsnn_pilot` compatibility probe.
It does **not** upgrade the probe to a production backend, and it does **not**
change the MAPLE `lsnn-v1` audited-domain gate.

## Scope

This record answers one narrow question:

> Does the local torch-only reconstruction reproduce the **public upstream
> LSNN energy graph** closely enough to justify the label
> `LSNN-conservative + GAFF compatibility`?

The reviewed answer is **yes for energies**, with a strict boundary:

- energy parity was observed at the `1e-5 kJ/mol` scale;
- this supports a **compatibility** label only;
- it does **not** inherit upstream force claims;
- it does **not** inherit the LSNN paper's free-energy accuracy claims.

This document also corrects one attribution error from an earlier note:

- the reviewed parity execution environment was
  `/home/axie/.cache/maple-envs/lsnn-benchmark-v2`;
- it was **not** the earlier `lsnn-pilot-venv` environment.

## Correct reviewer execution environment

The thread-supplied reviewer environment is:

```text
/home/axie/.cache/maple-envs/lsnn-benchmark-v2
```

Recorded versions:

```text
python 3.11.14
torch 2.12.0+cu130
torch_geometric 2.8.0.post1
torch_scatter 2.1.2+pt212cu130
numpy 2.4.6
openmm 8.5.2
```

Verified commands:

```bash
cat >/tmp/lsnn_benchmark_v2_python_version.py <<'PY'
import sys
print(sys.version.split()[0])
PY
/home/axie/.cache/maple-envs/lsnn-benchmark-v2/bin/python /tmp/lsnn_benchmark_v2_python_version.py

cat >/tmp/lsnn_benchmark_v2_versions.py <<'PY'
from importlib import metadata
for name in ['torch', 'torch-geometric', 'torch-scatter', 'numpy', 'openmm']:
    print(name, metadata.version(name))
PY
/home/axie/.cache/maple-envs/lsnn-benchmark-v2/bin/python /tmp/lsnn_benchmark_v2_versions.py
```

Observed output:

```text
3.11.14
torch 2.12.0+cu130
torch-geometric 2.8.0.post1
torch-scatter 2.1.2+pt212cu130
numpy 2.4.6
openmm 8.5.2
```

## Pinned inputs

### Common parity inputs

| Input | Pin |
| --- | --- |
| Upstream repo revision | `1768d068dcb1ea65e8585af3f4a0cbf4047d9125` |
| Upstream state dict | `Best_Trained_Models/280KDATASET2Kv3model.dict` |
| State-dict SHA256 | `5b7f9ec224f9264c0220e072ed917201e84e702bab62b331bade37c1ede3a83b` |
| Panel file | `docs/implicit-solvation/benchmarks/lsnn_pilot/panel.json` |
| Panel SHA256 | `0977a37b5c4efdabb3d7c474d365126b2b56847ee208fd585aeb8a3cdb451cbc` |
| FreeSolv `amber.tar.gz` SHA256 | `f1a72d5e2328b3a28dfee3c3590667b756a3ac9c0527e8ddf945044e664389a2` |
| FreeSolv `database.txt` SHA256 | `2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260` |

Verified command:

```bash
sha256sum \
  /home/axie/.cache/maple-benchmarks/LSNN-v1-1768d068/Best_Trained_Models/280KDATASET2Kv3model.dict \
  /home/axie/MAPLE/MAPLE-implicitsolv-route4/docs/implicit-solvation/benchmarks/lsnn_pilot/panel.json \
  /home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/vendor-audits/freesolv-current-20260726-v1/amber.tar.gz \
  /home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/vendor-audits/freesolv-current-20260726-v1/database.txt
```

Observed output:

```text
5b7f9ec224f9264c0220e072ed917201e84e702bab62b331bade37c1ede3a83b  /home/axie/.cache/maple-benchmarks/LSNN-v1-1768d068/Best_Trained_Models/280KDATASET2Kv3model.dict
0977a37b5c4efdabb3d7c474d365126b2b56847ee208fd585aeb8a3cdb451cbc  /home/axie/MAPLE/MAPLE-implicitsolv-route4/docs/implicit-solvation/benchmarks/lsnn_pilot/panel.json
f1a72d5e2328b3a28dfee3c3590667b756a3ac9c0527e8ddf945044e664389a2  /home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/vendor-audits/freesolv-current-20260726-v1/amber.tar.gz
2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260  /home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/vendor-audits/freesolv-current-20260726-v1/database.txt
```

### Single-molecule exact reproducibility inputs: methanol only

The thread supplied complete file identity only for the methanol exact parity
point:

| Input | Pin |
| --- | --- |
| Compound ID | `mobley_1636752` |
| Extracted `prmtop` SHA256 | `43d2740a3569ae1faf7edc8cb0bde1f9b959fc4b07656a5681024b2009f6c8eb` |
| Extracted `inpcrd` SHA256 | `dd0bf4e523c7e2393191424cf7213743495295c4c0ff5e2ce876df41d75db414` |

Verified command:

```bash
python - <<'PY'
from pathlib import Path
import tarfile, hashlib
amber = Path('/home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/vendor-audits/freesolv-current-20260726-v1/amber.tar.gz')
out = Path('/tmp/lsnn_doc_methanol')
out.mkdir(exist_ok=True)
with tarfile.open(amber, 'r:gz') as archive:
    for suffix in ('prmtop', 'inpcrd'):
        member = f'amber/mobley_1636752.{suffix}'
        path = out / f'mobley_1636752.{suffix}'
        path.write_bytes(archive.extractfile(member).read())
        print(suffix, hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

Observed output:

```text
prmtop 43d2740a3569ae1faf7edc8cb0bde1f9b959fc4b07656a5681024b2009f6c8eb
inpcrd dd0bf4e523c7e2393191424cf7213743495295c4c0ff5e2ce876df41d75db414
```

For methane and benzene, this note only records the thread-supplied summary
numbers below. It does **not** invent extra file hashes or a fake full command
transcript for those two molecules.

## Why this parity harness is scientifically legal

The public upstream repository still needs a narrow import shim for parity
inspection:

- `MachineLearning/GNN_Models.py:6` imports `torch_cluster`;
- the reviewed parity harness used `torch_cluster` as an **import-only stub**;
- the harness did **not** replace `torch_geometric.nn.MessagePassing`;
- the executed message-passing layers remained upstream:
  - `MachineLearning/GNN_Layers.py:13` `GBNeck_interaction(MessagePassing)`;
  - `MachineLearning/GNN_Layers.py:616` `IN_layer_all_swish_2pass(MessagePassing)`.

The specific upstream `run_multiple` path used for parity is also favorable for
small-molecule checking:

- `MachineLearning/GNN_Models.py:633` and `:642` pin
  `max_num_neighbors = 10000`;
- `MachineLearning/GNN_Models.py:675-676` prebuilds the long-range graph with
  `build_edge_idx(...)`;
- `MachineLearning/GNN_Models.py:785-810` shows that `build_edge_idx(...)`
  enumerates **all directed atom pairs**;
- `MachineLearning/GNN_Models.py:697-703` then derives the short-range GNN edge
  list by slicing that all-pairs graph with `edge_attributes < 0.6`.

So for methane, methanol, and benzene:

- the reviewed parity check does **not** depend on a truncated neighbor list;
- the `torch_cluster` import-only stub does **not** alter the executed path;
- the all-pairs graph plus `max_num_neighbors=10000` makes this a legitimate
  parity check for these tiny systems.

## Exact methanol parity point: complete executable recorder

The thread supplied enough exact information to freeze one **single-molecule
exact parity point** as an executable recorder. The block below computes the
local scalar exactly from the pinned MAPLE reconstruction, verifies the pinned
methanol files, and compares against the reviewer-supplied upstream scalar for
that same point.

This is the strongest fully reproducible single-molecule artifact available in
this branch note.

```bash
cat >/tmp/reviewer_methanol_parity_recorder.py <<'PY'
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, '/home/axie/MAPLE/MAPLE-implicitsolv-route4/docs/implicit-solvation/benchmarks/lsnn_pilot')
import run_lsnn_pilot as runner
from lsnn_model import load_lsnn_v1
import torch
from openmm import app

state = Path('/home/axie/.cache/maple-benchmarks/LSNN-v1-1768d068/Best_Trained_Models/280KDATASET2Kv3model.dict')
amber = Path('/home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/vendor-audits/freesolv-current-20260726-v1/amber.tar.gz')
out = Path('/tmp/reviewer-methanol-parity')
out.mkdir(exist_ok=True)
prmtop_path, inpcrd_path = runner._extract_amber_case(amber, 'mobley_1636752', out)
print('python', sys.version.split()[0])
print('state_sha256', hashlib.sha256(state.read_bytes()).hexdigest())
print('prmtop_sha256', hashlib.sha256(prmtop_path.read_bytes()).hexdigest())
print('inpcrd_sha256', hashlib.sha256(inpcrd_path.read_bytes()).hexdigest())
prmtop = app.AmberPrmtopFile(str(prmtop_path))
coords = app.AmberInpcrdFile(str(inpcrd_path))
system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
features = runner._gbn2_features(prmtop.topology, system)
model = load_lsnn_v1(str(state), torch.tensor(features, dtype=torch.float32)).eval()
positions = torch.tensor(coords.positions.value_in_unit(runner.unit.nanometer), dtype=torch.float32)
with torch.no_grad():
    local_zero = float(model(positions, torch.tensor(0.0), torch.tensor(0.0)).item())
    local_full = float(model(positions, torch.tensor(1.0), torch.tensor(1.0)).item())
upstream_full = -14.868452072
abs_diff = abs(upstream_full - local_full)
print('compound_id mobley_1636752')
print('local_lambda_0_0', f'{local_zero:.9f}')
print('upstream_lambda_1_1', f'{upstream_full:.9f}')
print('local_lambda_1_1', f'{local_full:.9f}')
print('abs_diff_kj_mol', f'{abs_diff:.11f}')
PY
/home/axie/.cache/maple-envs/lsnn-benchmark-v2/bin/python /tmp/reviewer_methanol_parity_recorder.py
```

Observed output:

```text
Warning on use of the timeseries module: If the inherent timescales of the system are long compared to those being analyzed, this statistical inefficiency may be an underestimate.  The estimate presumes the use of many statistically independent samples.  Tests should be performed to assess whether this condition is satisfied.   Be cautious in the interpretation of the data.

********* JAX NOT FOUND *********
 PyMBAR can run faster with JAX
 But will work fine without it
Either install with pip or conda:
      pip install pybar[jax]
               OR
      conda install pymbar
*********************************
Warning: importing 'simtk.openmm' is deprecated.  Import 'openmm' instead.
python 3.11.14
state_sha256 5b7f9ec224f9264c0220e072ed917201e84e702bab62b331bade37c1ede3a83b
prmtop_sha256 43d2740a3569ae1faf7edc8cb0bde1f9b959fc4b07656a5681024b2009f6c8eb
inpcrd_sha256 dd0bf4e523c7e2393191424cf7213743495295c4c0ff5e2ce876df41d75db414
compound_id mobley_1636752
local_lambda_0_0 0.000000000
upstream_lambda_1_1 -14.868452072
local_lambda_1_1 -14.868422508
abs_diff_kj_mol 0.00002956376
```

This exact recorder establishes three concrete facts for methanol:

- the decoupled endpoint is exactly `0.000000000 kJ/mol` locally;
- the local full-coupling endpoint is exactly `-14.868422508 kJ/mol`;
- the reviewer-supplied upstream endpoint differs by only
  `2.956376e-05 kJ/mol`.

## Cross-3-molecule extension evidence

Beyond the methanol exact recorder above, the thread also supplied a broader
cross-molecule summary:

```text
Across all 3 molecules and 6 lambda points, every absolute energy difference was <= 2.96e-5 kJ/mol.
Decoupled endpoints were exactly 0.
Intermediate/random lambda points stayed at about 1e-6 to 1e-5 kJ/mol absolute difference.
```

Endpoint examples supplied in-thread:

```text
methane  endpoint: upstream  4.779973984  vs local  4.779971123  (abs diff 2.86e-6 kJ/mol)
methanol endpoint: upstream -14.868452072 vs local -14.868422508 (abs diff 2.96e-5 kJ/mol)
benzene  endpoint: upstream -4.213361263  vs local -4.213356972  (abs diff 4.29e-6 kJ/mol)
```

This note treats that broader three-molecule statement as **extension evidence**,
not as a second fully reproducible artifact bundle, because this branch note was
not given the methane/benzene per-file hashes or their full executable command
transcripts.

## Why force parity is intentionally not claimed

The local model and the public upstream runtime do **not** expose the same force
convention.

### Upstream

In the public upstream runtime,
`MachineLearning/GNN_Models.py:436` enters:

- `with torch.no_grad():`
- `elec_energies = self.calculate_energies(...)`

The returned force is therefore not the conservative gradient of a single full
scalar energy surface that includes the electrostatic term in the usual way.

### Local reconstruction

The local reconstruction in `lsnn_model.py` instead exposes a single scalar
energy through `LSNNV1Energy` and lets OpenMM-Torch differentiate that full
scalar.

That difference is intentional and is why the pilot README and summary already
state:

- upstream force convention: detached electrostatics in the public runtime;
- local force convention: conservative full-energy gradient.

Therefore:

- **energy parity can be claimed** at compatibility scope;
- **force parity cannot be inherited** from the upstream runtime;
- **paper-level accuracy claims cannot be inherited** from this check alone.

## Additional scientific boundary

The vacuum Hamiltonian is still different from the paper workflow:

- this pilot uses archived FreeSolv `GAFF/AM1-BCC` Amber files;
- upstream `methods/LSNN.py` builds an OpenFF system with
  `SMIRNOFFTemplateGenerator`.

So even perfect energy parity between the two LSNN graph implementations would
still not turn this pilot into a strict reproduction of the published LSNN
free-energy workflow.

## Final label

The strongest justified label after this parity record is still:

```text
LSNN-conservative + GAFF compatibility
```

Not:

```text
upstream reproduction
production backend
paper-accuracy reproduction
```

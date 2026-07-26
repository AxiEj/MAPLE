# Route A research replica driver

## Periodic bulk-water Hamiltonian validation

Before starting any v3 packing replica, validate the candidate periodic
pure-water Hamiltonian with:

```bash
python examples/solvation/route_a/validate_bulk_water.py \
  --waterbox /path/to/hash-verified/waterbox.xyz \
  --checkpoint /path/to/MACE-OFF24_medium.model \
  --checkpoint-sha256 e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69 \
  --default-dtype float64 \
  --output /path/to/new/result \
  --device cuda
```

The source water box is pinned to a specific upstream commit and SHA256 in
`bulk_water.py`; the file is not vendored. The runner writes immutable,
hash-bound trajectory/RDF evidence and fails closed when its duration, sample
count or temperature gate is insufficient. Its default 10 ps production is a
research gate, not final liquid-water convergence or Route A accuracy
evidence. See
[`bulk-water-hamiltonian-validation.md`](../../../docs/solvation/route-a/bulk-water-hamiltonian-validation.md)
for the exact contract, first real preflight and open promotion gates.

The stress-aware NPT runner is separate:

```bash
python examples/solvation/route_a/validate_bulk_water_npt.py \
  --waterbox /path/to/hash-verified/waterbox.xyz \
  --waterbox-sha256 <sha256> \
  --expected-waters 64 \
  --checkpoint /path/to/MACE-OFF24_medium.model \
  --checkpoint-sha256 e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69 \
  --default-dtype float64 \
  --output /path/to/new/npt-result \
  --device cuda
```

It uses fixed-cell Langevin preconditioning followed by ASE isotropic MTK NPT.
The defaults reproduce the paper's 100 ps discarded interval and 400 ps
density-average duration, but the integrator is intentionally labeled
different from the paper's OpenMM Monte Carlo barostat. Three matching,
distinct-seed replicas are aggregated without pooling frames:

```bash
python examples/solvation/route_a/aggregate_bulk_water_npt_replicas.py \
  --output /path/to/new/npt-campaign.json \
  /path/to/npt-replica-01 \
  /path/to/npt-replica-02 \
  /path/to/npt-replica-03
```

The MACE cutoff is read from the loaded model, not accepted from the command
line, and numerical precision is explicit and hash-bound. Mixed float32 and
float64 replicas cannot be aggregated. Every NPT step retains the scalar
evidence needed to reproduce engineering gates. The campaign loader
revalidates every manifest, summary, raw NPZ hash, semantic array hash and the
exact preregistered protocol, then recomputes density, block SEM, drift,
temperature, pressure, duration, frame count, intermolecular RDFs, RDF block
SEM, O--O features and gates from the arrays. Distinct seeds must also have
distinct core trajectory hashes. The complete small-sample Student-t 95%
density interval must fit inside the fixed ±3% IAPWS band. The command writes
one non-overwriting, source-hash-bound JSON artifact; finite-size,
experimental-RDF and cross-engine gates remain mandatory.

## Fixed-occupancy development replica

`run_acetone_n1_replica.py` runs one real, nonperiodic MACE-OMOL
acetone + one-water alchemical development replica. It is intentionally not
wired to the public `#solvfe` result path because protocol v2 still requires
the pure-water packing term, conditioned multi-occupancy cycle, independent
replica agreement, and blind-holdout validation.

The driver writes:

- a hash-bound request;
- the complete immutable sampled coordinates and energy basis;
- MBAR, adjacent BAR, overlap, ESS, and half-trajectory diagnostics; and
- exact model, protocol, schedule, restraint, and move provenance.

It never overwrites a completed replica. The input sample archive must contain
`positions_angstrom` with 13 atoms ordered as acetone followed by one water.

Example:

```bash
python examples/solvation/route_a/run_acetone_n1_replica.py \
  --initial-samples /path/to/initial/samples.npz \
  --checkpoint /path/to/MACE-omol-0-extra-large-1024.model \
  --checkpoint-sha256 9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469 \
  --protocol docs/solvation/route-a/protocol-v2.json \
  --output /path/to/replica-01 \
  --seed 20260731
```

The output is a fixed-`n=1` development diagnostic, not a Route A absolute
hydration free energy and not evidence that Route A outperforms Route 2.
Protocol v3 supersedes this driver's v2 hard/soft conditioning interpretation;
the script is retained only to audit the completed sampler diagnostic and must
not be used to start new v2 replicas.

After at least three completed replicas, validate every artifact and apply the
protocol-frozen pairwise agreement gate:

```bash
python examples/solvation/route_a/aggregate_acetone_n1_replicas.py \
  --protocol docs/solvation/route-a/protocol-v2.json \
  --output /path/to/replica-aggregate.json \
  /path/to/replica-01 /path/to/replica-02 /path/to/replica-03
```

The aggregator refuses to pool failed, duplicate-seed, differently configured,
or protocol-mismatched replicas.

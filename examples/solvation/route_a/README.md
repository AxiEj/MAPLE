# Route A research replica driver

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

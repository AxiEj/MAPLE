# Hybrid MNSol accuracy boundary

Status: **pre-result infrastructure only**.  The total-SMD scalar, profile,
preregistration, and release tools are registered but disabled.  No hybrid
MNSol-10 prediction or score is claimed by this document.

The execution architecture passed a three-round verified-Pro audit only after
two fail-closed repair rounds.  The tracked audit is
`docs/route2/evidence/mace-mdp-polar-hybrid-mnsol10-pro-audit-v1.json`; its final
verdict is `APPROVE`, while the 148-row admission remains closed.

## Exact scalar

For fixed geometry `R` and solvent `s`, the solution-state scalar is

```text
Phi_s = E_vac^MACE-POLAR + G_hybrid-ddPCM + G_SMD-CDS .
```

The MNSol observable uses the same gas reference and therefore is

```text
DeltaG_pred = Phi_s - E_vac^MACE-POLAR
            = G_hybrid-ddPCM + G_SMD-CDS .
```

`AdditiveSolventOperationalLedger` accepts only `PySCFSMDCDSTerm`.  Adding the
CDS term changes the total-scalar and ledger identity, but does not change the
hybrid state equation, source/receiver spaces, root, or field-semantics hash.

## Evidence stages

Each stage has one responsibility and one output.

1. `prepare_mace_mdp_polar_hybrid_mnsol10_inputs.py`
   - opens captured user-supplied MNSol bytes;
   - validates the tracked protocol and frozen selection;
   - emits one private label-free coordinate/solvent bundle below `.omx`.
2. `seal_mace_mdp_polar_hybrid_mnsol10_fullsolv.py`
   - independently reconstructs the label-free bundle from source bytes;
   - binds the clean Git tree, preregistration, dataset manifests, checkpoints,
     source ledger, deterministic environment, and exact registered identity;
   - emits one content-addressed computation seal.
3. `run_mace_mdp_polar_hybrid_mnsol10_fullsolv.py`
   - starts with the BLAS/CUDA determinism environment already set;
   - opens no MNSol distribution or target;
   - claims the content-addressed execution before model evaluation;
   - emits exactly one complete or typed-failure prediction terminal.
4. `score_mace_mdp_polar_hybrid_mnsol10_fullsolv.py`
   - imports no model, continuum, coupling, Torch, MACE, or graph-longrange
     implementation;
   - revalidates seal, claim, input, prediction, source ledger, and exact MNSol
     archive identity before mechanically attaching labels;
   - emits one private row-level score bundle.
5. `publish_mace_mdp_polar_hybrid_mnsol10_public.py`
   - revalidates the whole prediction/scorer chain;
   - publishes only aggregate metrics and digests;
   - never emits opaque IDs, geometries, row predictions, targets, or errors.

Public projection is a retryable derivation from the private score.  It never
requires another model execution.

## Canonical artifact paths

```text
.omx/route2/hybrid-mnsol10/label-free-input-v1.json
.omx/route2/hybrid-mnsol10/seals/<execution_id>.json
~/.cache/maple-route2-stage-claims-v1/
  maple.route2.mnsol10-known-regression.attempt-slot.v1/execution-claim.json
.omx/route2/hybrid-mnsol10/runs/<execution_id>/prediction-terminal.json
.omx/route2/hybrid-mnsol10/runs/<execution_id>/private-score.json
docs/route2/evidence/mace-mdp-polar-hybrid-mnsol10-fullsolv-accuracy-v1.json
```

There are no alternate preregistration, prediction-output, or score-output
paths.  `attempt_slot_id` binds the scientific stage, exact dataset source and
selection, profile/scalar/state, checkpoints, frozen scientific contracts, and
exposure class; it deliberately excludes Git, code hashes, and environment
fingerprints.  The authoritative claim path is keyed by the immutable semantic
stage—not by raw protocol bytes or by `attempt_slot_id`—and is shared by every
checkout on the host.  Thus a formatting-only manifest change, new clean commit,
or worktree cannot regain retry rights after the stage is claimed.  Its custody
root comes from `getpwuid(getuid()).pw_dir`, not `HOME`; execution fails if
`HOME` differs, and each custody directory must be owned by that UID and not
group/world writable.  Canonical outputs use verified anonymous-inode,
no-replace publication with `fsync(file)` before `linkat` and `fsync(parent)`
afterward.  Every missing directory component is created with `mkdirat`, then
the parent receiving that child entry and the new directory are synchronized
before descent, so first-run directory creation cannot weaken claim durability.
This proves a locally sealed single attempt under honest host-user custody; it does
not claim globally unique execution across machines or protection from a host
owner who deletes evidence.  That stronger claim would require a pre-result
external append-only commitment.

## Frozen invocation

The final clean commit is executed in fresh processes with these variables set
before Python starts:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
```

Then, in order:

```bash
python tools/route2_release/prepare_mace_mdp_polar_hybrid_mnsol10_inputs.py \
  --source /private/path/MNSolDatabase_v2012.zip

python tools/route2_release/seal_mace_mdp_polar_hybrid_mnsol10_fullsolv.py \
  --source /private/path/MNSolDatabase_v2012.zip

python tools/route2_release/run_mace_mdp_polar_hybrid_mnsol10_fullsolv.py \
  --seal .omx/route2/hybrid-mnsol10/seals/<execution_id>.json

python tools/route2_release/score_mace_mdp_polar_hybrid_mnsol10_fullsolv.py \
  --source /private/path/MNSolDatabase_v2012.zip \
  --prediction \
  .omx/route2/hybrid-mnsol10/runs/<execution_id>/prediction-terminal.json

python tools/route2_release/publish_mace_mdp_polar_hybrid_mnsol10_public.py \
  --score .omx/route2/hybrid-mnsol10/runs/<execution_id>/private-score.json
```

If inspection finds a durable slot claim whose recorded Linux process no longer
exists and whose terminal is absent, close it without retrying the model:

```bash
python tools/route2_release/recover_mace_mdp_polar_hybrid_mnsol10_attempt.py \
  --seal .omx/route2/hybrid-mnsol10/seals/<execution_id>.json

python tools/route2_release/publish_mace_mdp_polar_hybrid_mnsol10_failure.py \
  --terminal \
  .omx/route2/hybrid-mnsol10/runs/<execution_id>/prediction-terminal.json
```

The exact seal path printed by the sealer replaces `<execution_id>`; it must not
be guessed or edited.

## Failure semantics

- Failure before the execution claim is a preflight/provenance error and has no
  scientific result.
- After the claim, provider, root, gate, candidate-record, and ordinary code
  failures produce a typed terminal with the already-validated record prefix.
- A failed numerical gate is negative evidence for the frozen profile; no row is
  removed and no setting is adapted.
- An attempt-slot claim forbids rerunning the same frozen scientific attempt
  under a new execution ID, Git tree, checkout, environment manifest, or filename.
- The claim binds Linux `boot_id`, PID, and `/proc` start ticks.  If a claimed
  process exits without a terminal—or recovery runs after a different boot—the
  verifier closes the slot as `aborted-unresolved`; the attempt remains
  unscorable and cannot be replaced.
- A publication failure after private scoring is recovered by rerunning only the
  aggregate publisher, never the model.

## Claim boundary

MNSol-10 has prior label and pure-profile exposure.  Even a pass establishes
only a known-panel regression for this exact fixed-geometry total scalar.  It
does not admit complete-distribution chemical accuracy, conformer transfer,
forces, Hessians, OPT/FREQ, MD, or Tier V.

The final accuracy gate remains the complete frozen 148-row confirmation
partition, executed as two independent label-free cold prediction processes and
scored only after their prediction digests are frozen.  The 102 records without
prior-pilot geometry overlap and the 46 overlap records must be reported
separately; neither stratum may be used to tune the frozen profile.

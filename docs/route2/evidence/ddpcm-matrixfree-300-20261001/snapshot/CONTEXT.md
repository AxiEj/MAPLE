# Exact streamed ddPCM exploratory milestone

**Initial context snapshot, superseded:** native Planner → Architect → Critic subsequently approved limited Gate 1–2 research; see `../../plans/ddpcm-matrixfree-300-native-reviews.md`. OS cgroup enforcement replaced sampling-only supervision. Current implementation/results/boundaries are in `REPORT.md`; the original constraints below remain, but initial capacity/pending status is historical.

User authorizes hands-on learning and iterative improvement toward >=300-atom implicit solvent without sacrificing accuracy.

## Constraints
- Preserve all tracked V2/V3 implementations and evidence byte-for-byte in this bounded prototype.
- FP64, lmax15, nleb1202, eta0.1; no FMM, fitting, radius changes, numerical production derivatives or tolerance relaxation.
- No public registration, expanded chemical-domain admission, GPU concurrency, commit or push.
- Native consensus planning was attempted but planner model is at capacity; adapted CLI preflight is unsupported. No consensus claim. Work remains independently authorized reversible research.

## Immediate experiment
Build exact streamed L/D/A/C actions and transposes with per-target geometry and bounded source tiles. Compare with untouched dense operators on small cases, then evaluate static/actual memory on larger continuum-only controls. Iterative E/adjoints are diagnostic only until stability and derivative qualification are reviewed.

## Target retained
>=300 atoms E/F/HVP/full analytic Hessian; prototype operator or energy success does NOT satisfy this target. Current domain16-500Da, CDS H/C/O and public entry remain unchanged.

## Hardware / operation envelope
Current host34GiB, GPU8GiB shared. CPU-only initial probes, no dense matrices beyond old limits; isolate long runs, sampled RSS cap8GiB and bounded walltime.

## Touchpoints
maple/solvation/continuum/ddpcm_response_operators.py
maple/solvation/continuum/response_topology.py
maple/solvation/continuum/torch_ddpcm.py
maple/solvation/experimental/mace_polar_response_core.py

## Unknowns
Exact direct-matvec time at300; conditioning/iteration counts; topology rejection frequency; MACE+CDS peak memory and mass-domain expansion; fully directional second derivatives.

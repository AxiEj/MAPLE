# Pure frozen CPU/CUDA non-MD examples

See [the versioned workflow guide](../../../docs/route2/PURE_MACE_POLAR_NONMD_V2.md)
for capability restrictions and the current real-run validation status.

These small water/NH3 inputs exercise one model potential, not reference
chemical accuracy. FREQ and IRC seeds and the two NH3 endpoints are copied
from the archived v1 real-checkpoint workflow records; OPT starts from a
small displacement of the archived optimized water geometry. They are not
new independently optimized or experimentally validated structures.

Run from the checkout root with `PYTHONPATH="$PWD"` and the pinned environment.
`cuda:N` selects the GPU for model inference; ddPCM/CDS remain CPU work.
To use CPU, change BOTH `#device` and the profile suffix to `nonmd-cpu-v2`.
NEB/CINEB images are pathway candidates, not certified saddles. Only a
converged, index-one-validated saddle can be interpreted as a model TS;
physical reaction assignment and barrier accuracy need independent evidence.

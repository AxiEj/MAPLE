# Torch solvation reference implementation: source notices

The legacy SMD-CDS coefficient algebra and analytical DAREAL surface port in
`maple/solvation/nonpolar/legacy_smd_cds.py` and
`maple/solvation/surfaces/legacy_dareal.py` are adapted from PySCF 2.13.1,
`pyscf/lib/solvent/mnsol.F`, commit
`f3754ed5baad778280dba5ad3f4982a8bed0ec2f`.

The upstream file attributes its Minnesota solvation implementation to
NWChem, and DAREAL to D. Liotard (December 1992), with derivatives by
D. Rinaldi and D. Liotard. The Torch port uses automatic differentiation
rather than copying the manual derivative routines. Changes include Python
and Torch expression of the energy algebra, explicit topology diagnostics,
and rejection of ambiguous radius-perturbation branches.

The PySCF distribution's Apache License 2.0 is reproduced in
[`PYSCF-APACHE-2.0.txt`](PYSCF-APACHE-2.0.txt). These adapted files retain
that source attribution; the repository's top-level license does not replace
the upstream license applicable to adapted material.

The ddPCM numerical equations and validation oracle are identified separately
in `maple/solvation/continuum/torch_ddpcm.py`: ddX 0.8.0 commit
`4d79e3d9caeae5e602683572a71cb550414f9b09` and the published ddPCM theory.
No ddX Fortran source is included in the production package.

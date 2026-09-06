# VQM24 Gate-A SG1 heavy-element preflight failure

The first bromine gas calculation stopped before its first SCF cycle because
PySCF 2.13.1 `sg1_prune` indexes an atomic-radius table of length 19 and cannot
handle Br (Z=35). No QM result or response was produced. The locked v1 input and
failure log are retained. Gate-A v2 uses the same geometries and probes but a
uniform PySCF level-3 nonlocal grid for every element; no result-dependent
molecule or field change was made.

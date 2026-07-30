# Frozen V1 QM induced-MEP raw record

`qm-induced-mep.json` is the unmodified output of the isolated, preregistered
PySCF finite-field helper.  It is retained so that the V1 report erratum can be
checked without repeating the 452.884-second QM calculation.

The companion `route2-v0-mace-mdp-induced-source-acetone-v1-execution-erratum.json`
records a bookkeeping defect in the original parent report: its QM-dipole
reproduction gate read the candidate response tensor instead of the independent
QM tensor.  The independent source-MEP rejection is unaffected: the raw MEP
errors remain above their preregistered limits.  No source parameter, QM input,
experimental solvation label, or decision threshold was changed.

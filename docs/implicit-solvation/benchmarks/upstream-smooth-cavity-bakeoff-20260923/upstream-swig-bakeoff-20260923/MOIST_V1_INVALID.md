# MOIST area wrapper v1 — invalid execution, not a method result

The preregistered v1 wrapper passed `atoms.positions.T / Bohr` into the
upstream `moist.Structure` Python API. Current MOIST main expects array shape
`(natoms, 3)` and itself maps it to native Fortran `(3, natoms)`. The other
seven non-three-atom cases failed shape validation. Three-atom cases such as
water were silently transposed because their shape stayed `(3, 3)`, so their
areas/rotations were calculated for the wrong geometry. All v1 MOIST result
numbers are invalid as model evidence. Preserve its protocol, script, output,
and log; do not overwrite or selectively keep its water result. The corrected
v2 wrapper adds exact cavity-center readback before interpreting any area.

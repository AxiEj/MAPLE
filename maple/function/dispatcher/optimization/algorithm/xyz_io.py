# -*- coding: utf-8 -*-
"""Re-export of the shared extxyz writer.

The canonical implementation lives at
maple.function.utility.xyz_io, so SCAN / TS / IRC can share the same
function without reaching into the optimization subtree.
"""
from maple.function.utility.xyz_io import write_xyz

__all__ = ["write_xyz"]

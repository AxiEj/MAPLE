"""Shared DCD binary-format constants and header helpers."""

import struct

import numpy as np


_DCD_HEADER_SIZE = 84
_DCD_TITLE_BLOCK_SIZE = 160
_DCD_CORD_MAGIC = 84


def _dcd_delta_fs(header_data: bytes) -> float:
    """Decode the DCD DELTA field as picoseconds and return femtoseconds."""
    delta_ps = struct.unpack_from('<f', header_data, 9 * 4)[0]
    if np.isfinite(delta_ps) and delta_ps >= 1e-12:
        return float(delta_ps) * 1000.0

    # Backward compatibility for old MAPLE DCD files that incorrectly wrote
    # DELTA as an integer number of picoseconds.
    legacy_delta_ps = np.frombuffer(header_data, dtype=np.int32)[9]
    return float(legacy_delta_ps) * 1000.0

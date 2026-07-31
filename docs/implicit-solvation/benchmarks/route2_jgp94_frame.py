"""Compatibility import for the promoted Route-2 JGP94 frame library.

The implementation lives in ``maple.function.calculator.extra_correction`` so
force-profile code and benchmark canaries share exactly one mathematical
transform.  This shim preserves the historical benchmark import path.
"""

from maple.function.calculator.extra_correction.implicit.route2_body_frame import (
    JGP94Frame,
    JGP94FrameVJP,
    body_dipoles,
    build_jgp94_frame,
    jgp94_frame_vjp,
)

__all__ = [
    "JGP94Frame",
    "JGP94FrameVJP",
    "body_dipoles",
    "build_jgp94_frame",
    "jgp94_frame_vjp",
]

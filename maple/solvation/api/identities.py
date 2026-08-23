"""Dependency-light shared identities for Route-2 public contracts.

An identity used by both an implementation and its registry entry must have
one authoritative definition.  Keeping these strings here avoids importing a
numerical continuum module from the dependency-light API registry and avoids
duplicating literals across layers.
"""

MACE_MDP_POLAR_HYBRID_HARMONIC_DDPCM_COUPLING_ID = (
    "route2-coupling-macemdppoint-macepolarinduced-gto1p5-"
    "smoothharmonic-ddpcm-nativefield8-v2"
)


__all__ = ["MACE_MDP_POLAR_HYBRID_HARMONIC_DDPCM_COUPLING_ID"]

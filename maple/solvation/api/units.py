"""Single public unit declaration for Route-2 ASE results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnitContract:
    """Units exposed at the public ASE boundary.

    ``A`` is the machine-readable ASCII spelling of Å used in manifests.
    Internal backends must convert to this contract before constructing a
    public result.
    """

    energy: str
    forces: str
    hessian: str
    coordinates: str

    def __post_init__(self) -> None:
        for name in ("energy", "forces", "hessian", "coordinates"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Unit {name!r} must be a non-empty string.")


ASE_PUBLIC_UNITS = UnitContract(
    energy="eV",
    forces="eV/A",
    hessian="eV/A^2",
    coordinates="A",
)

__all__ = ["ASE_PUBLIC_UNITS", "UnitContract"]

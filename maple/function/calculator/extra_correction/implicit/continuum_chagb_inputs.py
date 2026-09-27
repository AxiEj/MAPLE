"""Immutable fixed-charge inputs for a separate CHA continuum research profile.

No solvent energy, force, atom typing, charge generation, or Amber executable is
provided here. A source charge vector and its serialized native representation
remain distinct; the latter is the parity input for the historical endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from torch import Tensor


PROFILE = "chagb-r6-pbsa-continuum-v1"
MAX_SERIALIZATION_DRIFT_E = 1.0e-7
CHARGE_SUM_TOLERANCE_E = 1.0e-7
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "atom_ids",
        "atom_names",
        "elements",
        "atomic_numbers",
        "gaff2_types",
        "bonds",
        "declared_charge_e",
        "source_charges_e",
        "effective_charges_e",
        "source_charges_sha256",
        "effective_charges_sha256",
        "source_mol2_sha256",
        "prepared_prmtop_sha256",
        "parameter_source_sha256",
        "serialization_profile",
        "bondi_radii_angstrom",
        "cha_radii_angstrom",
        "lj_rmin_angstrom",
        "lj_epsilon_kcal_mol",
        "content_sha256",
    }
)


def _canonical(value, *, newline: bool = False) -> bytes:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return (serialized + ("\n" if newline else "")).encode("utf-8")


def _sha(value, *, newline: bool = False) -> str:
    return hashlib.sha256(_canonical(value, newline=newline)).hexdigest()


def _coordinate_sha(topology_sha256: str, atom_ids: tuple[int, ...], positions) -> str:
    return _sha(
        {
            "topology_sha256": topology_sha256,
            "atom_ids": list(atom_ids),
            "positions_angstrom": positions.detach().cpu().tolist(),
        }
    )


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _strings(name: str, value: object, count: int) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != count
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{name} requires {count} nonempty strings.")
    return tuple(value)


def _numbers(
    name: str, value: object, count: int, *, strictly_positive: bool = False
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{name} requires exactly {count} numeric values.")
    if any(
        isinstance(item, bool) or not isinstance(item, (float, int)) for item in value
    ):
        raise TypeError(f"{name} values must be numbers, not booleans or strings.")
    numbers = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in numbers):
        raise ValueError(f"{name} must be finite.")
    if strictly_positive and any(item <= 0.0 for item in numbers):
        raise ValueError(f"{name} must be strictly positive.")
    return numbers


@dataclass(frozen=True)
class ContinuumChaTensors:
    """Fixed parameter tensors, separate from every coordinate graph."""

    topology_sha256: str
    content_sha256: str
    charges_e: Tensor
    bondi_radii_angstrom: Tensor
    cha_radii_angstrom: Tensor
    lj_rmin_angstrom: Tensor
    lj_epsilon_kcal_mol: Tensor

    def assert_current(self) -> None:
        """Reject in-place mutation of any nominally fixed parameter tensor."""
        import torch

        values = {
            "charges_e": self.charges_e,
            "bondi_radii_angstrom": self.bondi_radii_angstrom,
            "cha_radii_angstrom": self.cha_radii_angstrom,
            "lj_rmin_angstrom": self.lj_rmin_angstrom,
            "lj_epsilon_kcal_mol": self.lj_epsilon_kcal_mol,
        }
        count = len(self.charges_e)
        device = self.charges_e.device
        if any(
            not isinstance(item, torch.Tensor)
            or item.dtype != torch.float64
            or item.shape != (count,)
            or item.device != device
            or item.requires_grad
            or not bool(torch.isfinite(item).all())
            for item in values.values()
        ):
            raise ValueError("CHA parameter tensor identity or shape changed.")
        if (
            _sha(
                {
                    "topology_sha256": self.topology_sha256,
                    **{
                        name: item.detach().cpu().tolist()
                        for name, item in values.items()
                    },
                }
            )
            != self.content_sha256
        ):
            raise ValueError("CHA parameter tensor hash no longer matches its values.")


@dataclass(frozen=True)
class ContinuumChaTopology:
    """Versioned source/effective charge and per-site parameter identity."""

    content_sha256: str
    source_mol2_sha256: str
    prepared_prmtop_sha256: str
    parameter_source_sha256: str
    serialization_profile: str
    atom_ids: tuple[int, ...]
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    gaff2_types: tuple[str, ...]
    bonds: tuple[tuple[int, int, str], ...]
    declared_charge_e: int
    source_charges_e: tuple[float, ...]
    effective_charges_e: tuple[float, ...]
    source_charges_sha256: str
    effective_charges_sha256: str
    bondi_radii_angstrom: tuple[float, ...]
    cha_radii_angstrom: tuple[float, ...]
    lj_rmin_angstrom: tuple[float, ...]
    lj_epsilon_kcal_mol: tuple[float, ...]

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "profile": PROFILE,
            **{
                name: getattr(self, name)
                for name in _FIELDS - {"schema_version", "profile", "content_sha256"}
            },
        }

    def __post_init__(self) -> None:
        # A frozen dataclass alone does not freeze its invariant: direct
        # construction or dataclasses.replace must not bypass the loader.
        self.assert_current(self.content_sha256)

    def assert_current(self, expected_content_sha256: str) -> None:
        """Recheck internal values against the caller's external content pin."""
        expected = _digest("expected_content_sha256", expected_content_sha256)
        self._validate_semantics()
        if self.content_sha256 != expected or _sha(self._payload()) != expected:
            raise ValueError("CHA topology hash differs from its external identity.")

    def _validate_semantics(self) -> None:
        """Validate every topology invariant for all construction paths."""
        for name in (
            "content_sha256",
            "source_mol2_sha256",
            "prepared_prmtop_sha256",
            "parameter_source_sha256",
            "source_charges_sha256",
            "effective_charges_sha256",
        ):
            _digest(name, getattr(self, name))

        count = len(self.atom_ids)
        if (
            not isinstance(self.atom_ids, tuple)
            or count == 0
            or any(type(atom_id) is not int for atom_id in self.atom_ids)
            or self.atom_ids != tuple(range(1, count + 1))
        ):
            raise ValueError("atom_ids must preserve contiguous source atom order.")
        vector_fields = (
            "atom_names",
            "elements",
            "atomic_numbers",
            "gaff2_types",
            "source_charges_e",
            "effective_charges_e",
            "bondi_radii_angstrom",
            "cha_radii_angstrom",
            "lj_rmin_angstrom",
            "lj_epsilon_kcal_mol",
        )
        if any(not isinstance(getattr(self, name), tuple) for name in vector_fields):
            raise ValueError("CHA topology vector fields must be immutable tuples.")
        _strings("atom_names", self.atom_names, count)
        _strings("elements", self.elements, count)
        _strings("gaff2_types", self.gaff2_types, count)
        if (
            not isinstance(self.atomic_numbers, tuple)
            or len(self.atomic_numbers) != count
            or any(type(number) is not int for number in self.atomic_numbers)
        ):
            raise ValueError("atomic_numbers must contain one integer per atom.")
        from ase.data import chemical_symbols

        if any(
            not (1 <= number < len(chemical_symbols))
            or chemical_symbols[number] != symbol
            for number, symbol in zip(self.atomic_numbers, self.elements)
        ):
            raise ValueError("Atomic numbers and element symbols disagree.")

        if not isinstance(self.bonds, tuple):
            raise ValueError("bonds must be an immutable sequence of typed bonds.")
        for bond in self.bonds:
            if (
                not isinstance(bond, tuple)
                or len(bond) != 3
                or any(type(index) is not int for index in bond[:2])
                or not (0 <= bond[0] < bond[1] < count)
                or not isinstance(bond[2], str)
                or not bond[2].strip()
            ):
                raise ValueError("Invalid or reordered bond in CHA topology.")
        if len({bond[:2] for bond in self.bonds}) != len(self.bonds):
            raise ValueError("Duplicate bond in CHA topology.")

        if type(self.declared_charge_e) is not int:
            raise TypeError("declared_charge_e must be an integer.")
        source = _numbers("source_charges_e", self.source_charges_e, count)
        effective = _numbers("effective_charges_e", self.effective_charges_e, count)
        if (
            _sha(list(source), newline=True) != self.source_charges_sha256
            or _sha(list(effective), newline=True) != self.effective_charges_sha256
        ):
            raise ValueError("CHA topology charge-vector hash mismatch.")
        for name in (
            "bondi_radii_angstrom",
            "cha_radii_angstrom",
            "lj_rmin_angstrom",
            "lj_epsilon_kcal_mol",
        ):
            _numbers(name, getattr(self, name), count, strictly_positive=True)

        if self.serialization_profile not in {
            "ambertools26-prmtop-charge-5e16.8-v1",
            "direct-fixed-charge-v1",
        }:
            raise ValueError("Unknown fixed-charge serialization profile.")
        if self.serialization_profile == "direct-fixed-charge-v1" and (
            source != effective
        ):
            raise ValueError(
                "Direct fixed-charge inputs must not change their charge vector."
            )
        if (
            abs(sum(source) - self.declared_charge_e) > CHARGE_SUM_TOLERANCE_E
            or abs(sum(effective) - self.declared_charge_e) > CHARGE_SUM_TOLERANCE_E
            or max(abs(before - after) for before, after in zip(source, effective))
            > MAX_SERIALIZATION_DRIFT_E
        ):
            raise ValueError("CHA topology fixed charge identity changed.")

    @property
    def atom_count(self) -> int:
        return len(self.atom_ids)

    @classmethod
    def from_mapping(
        cls,
        artifact: Mapping[str, object],
        *,
        expected_content_sha256: str,
        expected_source_charge_sha256: str | None = None,
        expected_source_mol2_sha256: str | None = None,
    ) -> ContinuumChaTopology:
        if not isinstance(artifact, Mapping) or set(artifact) != _FIELDS:
            raise ValueError("CHA topology artifact fields are missing or unexpected.")
        if (
            type(artifact["schema_version"]) is not int
            or artifact["schema_version"] != 1
            or artifact["profile"] != PROFILE
        ):
            raise ValueError("Unsupported CHA topology schema or profile.")
        content_sha256 = _digest("content_sha256", artifact["content_sha256"])
        if content_sha256 != _digest(
            "expected_content_sha256", expected_content_sha256
        ):
            raise ValueError("CHA topology differs from the external content pin.")
        if (
            _sha(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "content_sha256"
                }
            )
            != content_sha256
        ):
            raise ValueError("CHA topology artifact content hash mismatch.")

        topology = cls(
            content_sha256=content_sha256,
            source_mol2_sha256=artifact["source_mol2_sha256"],
            prepared_prmtop_sha256=artifact["prepared_prmtop_sha256"],
            parameter_source_sha256=artifact["parameter_source_sha256"],
            serialization_profile=artifact["serialization_profile"],
            atom_ids=tuple(artifact["atom_ids"]),
            atom_names=tuple(artifact["atom_names"]),
            elements=tuple(artifact["elements"]),
            atomic_numbers=tuple(artifact["atomic_numbers"]),
            gaff2_types=tuple(artifact["gaff2_types"]),
            bonds=tuple(tuple(bond) for bond in artifact["bonds"]),
            declared_charge_e=artifact["declared_charge_e"],
            source_charges_e=tuple(
                float(value) for value in artifact["source_charges_e"]
            ),
            effective_charges_e=tuple(
                float(value) for value in artifact["effective_charges_e"]
            ),
            source_charges_sha256=artifact["source_charges_sha256"],
            effective_charges_sha256=artifact["effective_charges_sha256"],
            bondi_radii_angstrom=tuple(
                float(value) for value in artifact["bondi_radii_angstrom"]
            ),
            cha_radii_angstrom=tuple(
                float(value) for value in artifact["cha_radii_angstrom"]
            ),
            lj_rmin_angstrom=tuple(
                float(value) for value in artifact["lj_rmin_angstrom"]
            ),
            lj_epsilon_kcal_mol=tuple(
                float(value) for value in artifact["lj_epsilon_kcal_mol"]
            ),
        )
        if expected_source_charge_sha256 is not None and (
            topology.source_charges_sha256
            != _digest("expected_source_charge_sha256", expected_source_charge_sha256)
        ):
            raise ValueError("Fixed source charge identity differs from caller state.")
        if expected_source_mol2_sha256 is not None and (
            topology.source_mol2_sha256
            != _digest("expected_source_mol2_sha256", expected_source_mol2_sha256)
        ):
            raise ValueError("Source MOL2 identity differs from caller state.")
        return topology

    def tensors(
        self, *, expected_content_sha256: str, device: str = "cpu"
    ) -> ContinuumChaTensors:
        import torch

        self.assert_current(expected_content_sha256)

        def fixed(values):
            return torch.tensor(values, dtype=torch.float64, device=device)

        values = ContinuumChaTensors(
            topology_sha256=self.content_sha256,
            content_sha256=_sha(
                {
                    "topology_sha256": self.content_sha256,
                    "charges_e": self.effective_charges_e,
                    "bondi_radii_angstrom": self.bondi_radii_angstrom,
                    "cha_radii_angstrom": self.cha_radii_angstrom,
                    "lj_rmin_angstrom": self.lj_rmin_angstrom,
                    "lj_epsilon_kcal_mol": self.lj_epsilon_kcal_mol,
                }
            ),
            charges_e=fixed(self.effective_charges_e),
            bondi_radii_angstrom=fixed(self.bondi_radii_angstrom),
            cha_radii_angstrom=fixed(self.cha_radii_angstrom),
            lj_rmin_angstrom=fixed(self.lj_rmin_angstrom),
            lj_epsilon_kcal_mol=fixed(self.lj_epsilon_kcal_mol),
        )
        values.assert_current()
        return values


@dataclass(frozen=True)
class ContinuumChaCoordinates:
    """One dynamic geometry, preserving the caller's coordinate tensor graph."""

    positions_angstrom: Tensor
    atom_ids: tuple[int, ...]
    topology_sha256: str
    content_sha256: str

    @classmethod
    def from_tensor(
        cls,
        positions_angstrom: Tensor,
        topology: ContinuumChaTopology,
        *,
        atom_ids: tuple[int, ...] | None = None,
    ) -> ContinuumChaCoordinates:
        import torch

        if (
            not isinstance(positions_angstrom, torch.Tensor)
            or positions_angstrom.dtype != torch.float64
        ):
            raise TypeError("CHA coordinates must be an explicit torch.float64 tensor.")
        if tuple(positions_angstrom.shape) != (topology.atom_count, 3) or not bool(
            torch.isfinite(positions_angstrom).all()
        ):
            raise ValueError("CHA coordinates must be finite with shape (N, 3).")
        ordered = topology.atom_ids if atom_ids is None else tuple(atom_ids)
        if ordered != topology.atom_ids:
            raise ValueError("Coordinate atom IDs differ from frozen topology order.")
        # Clone keeps the autograd edge but prevents the caller from changing a
        # prepared geometry in-place. The wrapper itself is checked at use time.
        owned = positions_angstrom.clone()
        fingerprint = _coordinate_sha(topology.content_sha256, ordered, owned)
        return cls(owned, ordered, topology.content_sha256, fingerprint)

    def assert_current(self) -> None:
        """Fail if the public tensor was mutated after its identity was frozen."""
        import torch

        if not bool(torch.isfinite(self.positions_angstrom).all()) or (
            _coordinate_sha(
                self.topology_sha256, self.atom_ids, self.positions_angstrom
            )
            != self.content_sha256
        ):
            raise ValueError("CHA coordinate hash no longer matches its tensor.")

"""Coordinate-connected zero-field MACE-POLAR Torch boundary.

This module is deliberately a narrow adapter around the release-bound legacy
model adapter.  It neither reimplements MACE inference nor changes the frozen
checkpoint, source basis, long-range evaluator, or zero-field semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

_NEIGHBOR_CUTOFF_GUARD_ANGSTROM = 1.0e-8


class _TorchModelOnlyMACEPolCalculator(MACEPolCalculator):
    """V3-only live-coordinate boundary; the historical calculator is untouched."""

    def implicit_solv_init(self, implicit: str, solvent: str) -> None:
        del implicit, solvent
        self.solvent_correction = None

    def _batch_dict(
        self,
        atoms,
        *,
        positions_tensor: Any = None,
    ) -> dict[str, Any]:
        if positions_tensor is None:
            return super()._batch_dict(atoms)
        torch = __import__("torch")
        batch = self._mace._atoms_to_batch(self._atoms_for_mace(atoms))
        model_parameter = next(self.model.parameters())
        model_dtype = model_parameter.dtype
        model_device = model_parameter.device
        result = batch.to_dict()
        for key, value in tuple(result.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                result[key] = value.to(dtype=model_dtype)
        if not torch.is_tensor(positions_tensor):
            raise TypeError("positions_tensor must be a torch tensor.")
        if positions_tensor.shape != (len(atoms), 3):
            raise ValueError(
                "positions_tensor must have shape "
                f"({len(atoms)}, 3); received {tuple(positions_tensor.shape)}."
            )
        if positions_tensor.dtype != model_dtype:
            raise TypeError(
                "positions_tensor must use the model dtype "
                f"{model_dtype}; received {positions_tensor.dtype}."
            )
        if positions_tensor.device != model_device:
            raise ValueError(
                "positions_tensor must be on the model device "
                f"{model_device}; received {positions_tensor.device}."
            )
        if not bool(torch.isfinite(positions_tensor).all()):
            raise ValueError("positions_tensor must contain only finite values.")
        reference_positions = torch.as_tensor(
            atoms.get_positions(), dtype=model_dtype, device=model_device
        )
        if not torch.equal(positions_tensor.detach(), reference_positions):
            raise ValueError(
                "positions_tensor must exactly represent the supplied central "
                "geometry; changed coordinates require rebuilding and certifying "
                "the MACE neighbor topology before analytic differentiation."
            )
        result["positions"] = positions_tensor
        return self._long_range_evaluator.prepare_batch(result, r_max=self.r_max)

    def zero_field_energy_source_torch(
        self, atoms, positions_tensor: Any
    ) -> tuple[Any, Any]:
        """Return the exact existing zero-field energy/source on a live R graph."""
        torch = __import__("torch")
        batch = self._batch_dict(atoms, positions_tensor=positions_tensor)
        output = self._model_forward(
            batch,
            compute_force=False,
            compute_stress=False,
            compute_hessian=False,
        )
        if not isinstance(output, dict):
            raise TypeError("MACE-POLAR zero-field forward must return a dictionary.")
        energy = output.get("energy")
        density = output.get("density_coefficients")
        if not torch.is_tensor(energy):
            raise RuntimeError(
                "MACE-POLAR zero-field forward returned no energy tensor."
            )
        energy = energy.sum()
        if (
            energy.ndim != 0
            or energy.dtype != self.dtype
            or (energy.device != positions_tensor.device)
        ):
            raise RuntimeError(
                "MACE-POLAR zero-field energy must be one scalar tensor on the "
                "configured device and dtype."
            )
        expected_density_shape = (len(atoms), 4)
        if (
            not torch.is_tensor(density)
            or density.shape != expected_density_shape
            or density.dtype != self.dtype
            or density.device != positions_tensor.device
        ):
            received = None if density is None else tuple(density.shape)
            raise RuntimeError(
                "MACE-POLAR zero-field raw density must have shape "
                f"{expected_density_shape} on the configured device and dtype; "
                f"received {received}."
            )
        if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(density).all()):
            raise RuntimeError("MACE-POLAR zero-field energy/source is non-finite.")
        if positions_tensor.requires_grad and (
            not energy.requires_grad or not density.requires_grad
        ):
            raise RuntimeError(
                "MACE-POLAR zero-field energy/source is disconnected from the "
                "caller-owned coordinate graph."
            )
        return energy, density


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class MACEPolarTorchGraphAdapter:
    """Expose the official zero-field vacuum energy/source on one live graph."""

    __slots__ = (
        "_base",
        "_calculator",
        "_configuration_sha256",
        "_model_metadata",
        "_sealed",
        "device",
        "dtype",
        "provider_id",
        "model_profile_id",
        "provenance",
        "provenance_sha256",
        "domain",
        "release_contract",
    )

    def __init__(self, base: object) -> None:
        configuration = getattr(base, "configuration_sha256", None)
        calculator = getattr(base, "_calculator", None)
        if not callable(configuration):
            raise TypeError(
                "MACE-POLAR Torch graph requires a configured base adapter."
            )
        if not callable(getattr(calculator, "zero_field_energy_source_torch", None)):
            raise TypeError(
                "MACE-POLAR calculator lacks zero_field_energy_source_torch()."
            )
        domain = getattr(base, "domain", None)
        if not callable(getattr(domain, "validate_atoms", None)):
            raise TypeError("MACE-POLAR Torch graph requires the base model domain.")
        required_attributes = (
            "device",
            "dtype",
            "provider_id",
            "model_profile_id",
            "provenance",
            "provenance_sha256",
            "release_contract",
        )
        missing = [name for name in required_attributes if not hasattr(base, name)]
        if missing:
            raise TypeError(
                "MACE-POLAR Torch graph base adapter is missing: " + ", ".join(missing)
            )
        base_configuration = configuration()
        neighbor_cutoff_angstrom = float(getattr(calculator, "r_max", math.nan))
        if not math.isfinite(neighbor_cutoff_angstrom) or (
            neighbor_cutoff_angstrom <= 0.0
        ):
            raise ValueError(
                "MACE-POLAR Torch graph requires a finite positive model cutoff."
            )
        metadata = {
            "schema": "route2-mace-polar-zero-field-torch-graph-v1",
            "provider_id": base.provider_id,
            "model_profile_id": base.model_profile_id,
            "provenance_sha256": base.provenance_sha256,
            "base_configuration_sha256": base_configuration,
            "device": str(base.device),
            "dtype": str(base.dtype),
            "checkpoint_sha256": base.release_contract.checkpoint_sha256,
            "long_range_evaluator_profile": (
                base.release_contract.long_range_evaluator_profile
            ),
            "field_state": "frozen-zero-external-field",
            "source_basis": "raw-mace-polar-density-coefficients-[q,y,z,x]",
            "source_units": "checkpoint-native-physical-multipoles",
            "neighbor_topology_policy": (
                "all-nonself-pairs-fixed-mask-conservative-guard-v1"
            ),
            "neighbor_cutoff_angstrom": neighbor_cutoff_angstrom,
            "neighbor_cutoff_guard_angstrom": _NEIGHBOR_CUTOFF_GUARD_ANGSTROM,
        }
        graph_configuration = _hash(
            {
                **metadata,
                "implementation_sha256": _implementation_sha256(),
            }
        )
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_calculator", calculator)
        object.__setattr__(self, "_configuration_sha256", graph_configuration)
        object.__setattr__(self, "_model_metadata", MappingProxyType(metadata))
        for name in required_attributes:
            object.__setattr__(self, name, getattr(base, name))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarTorchGraphAdapter is immutable.")
        object.__setattr__(self, name, value)

    @property
    def model_metadata(self) -> Mapping[str, object]:
        return self._model_metadata

    def metadata(self) -> dict[str, object]:
        return dict(self._model_metadata)

    def _current_configuration_sha256(self) -> str:
        metadata = dict(self._model_metadata)
        metadata["base_configuration_sha256"] = self._base.configuration_sha256()
        return _hash(
            {
                **metadata,
                "implementation_sha256": _implementation_sha256(),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise ValueError("MACE-POLAR zero-field Torch graph configuration drifted.")
        return current

    def topology_diagnostics(self, atoms: object) -> dict[str, object]:
        """Certify one conservative fixed MACE neighbor topology.

        The MACE cutoff polynomial may be smoother than this guard requires;
        this certificate intentionally makes no global regularity claim.  It
        only rejects central geometries at coincident-center singularities or
        too close to the discrete neighbor-list cutoff for a local Hessian.
        """

        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if positions.shape != (len(atoms), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("MACE-POLAR topology positions must be finite (N,3).")
        cutoff = float(self._model_metadata["neighbor_cutoff_angstrom"])
        pairs: list[list[object]] = []
        distances: list[float] = []
        cutoff_distances: list[float] = []
        for first in range(len(atoms)):
            for second in range(first + 1, len(atoms)):
                distance = float(np.linalg.norm(positions[second] - positions[first]))
                if not math.isfinite(distance):
                    raise ValueError("MACE-POLAR pair distance is non-finite.")
                if distance == 0.0:
                    raise ValueError(
                        "MACE-POLAR analytic topology rejects coincident atom centers."
                    )
                cutoff_distance = abs(distance - cutoff)
                if cutoff_distance <= _NEIGHBOR_CUTOFF_GUARD_ANGSTROM:
                    raise ValueError(
                        "MACE-POLAR analytic topology is within the conservative "
                        "neighbor-cutoff guard."
                    )
                inside = distance < cutoff
                pairs.append([first, second, inside])
                distances.append(distance)
                cutoff_distances.append(cutoff_distance)
        pair_mask_order_sha256 = _hash(
            {
                "schema": "route2-mace-polar-neighbor-pair-mask-order-v1",
                "atom_count": len(atoms),
                "cutoff_angstrom": cutoff,
                "ordered_pairs": pairs,
            }
        )
        return {
            "schema": "route2-mace-polar-neighbor-topology-diagnostics-v1",
            "policy": self._model_metadata["neighbor_topology_policy"],
            "atom_count": len(atoms),
            "nonself_pair_count": len(pairs),
            "neighbor_pair_count": sum(bool(pair[2]) for pair in pairs),
            "cutoff_angstrom": cutoff,
            "cutoff_guard_angstrom": _NEIGHBOR_CUTOFF_GUARD_ANGSTROM,
            "minimum_pair_distance_angstrom": (min(distances) if distances else None),
            "minimum_cutoff_distance_angstrom": (
                min(cutoff_distances) if cutoff_distances else None
            ),
            "pair_mask_order_sha256": pair_mask_order_sha256,
            "locally_fixed_neighbor_mask": True,
            "global_regularity_proved": False,
        }

    def energy_source_torch(self, atoms: object, positions_angstrom):
        """Return scalar vacuum energy (eV) and raw ``(N,4)`` source tensor."""

        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        self.topology_diagnostics(atoms)
        return self._calculator.zero_field_energy_source_torch(
            atoms,
            positions_angstrom,
        )


__all__ = ["MACEPolarTorchGraphAdapter"]

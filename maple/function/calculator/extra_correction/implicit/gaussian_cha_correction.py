"""Opt-in programmatic correction adapter for Gaussian-sign continuum CHA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .common import KJ_PER_MOL_PER_HARTREE
from .continuum_chagb_inputs import ContinuumChaTopology
from .result import SolvationResult
from .torch_chagb_gaussian import (
    GAUSSIAN_CHA_MODEL_IDENTITY,
    validate_gaussian_sigma_e,
)
from .torch_continuum_chagb import MAX_QUADRATURE_ORDER
from .torch_continuum_chagb_gaussian import continuum_gaussian_cha_scalar

KCAL_PER_MOL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184
_ATOM_IDENTITY_ARRAY = "_maple_implicit_atom_identity"


def _strict_integer_tuple(name, values, *, count: int) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.shape != (count,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"Current {name} must be an exact integer vector of length {count}."
        )
    return tuple(array.tolist())


@dataclass(frozen=True, init=False)
class GaussianChaCorrection:
    """Immutable topology-bound energy/force correction for one model width."""

    topology: ContinuumChaTopology
    expected_topology_sha256: str
    sigma_e: float
    order: int
    _frozen_atom_identity: tuple[int, ...]
    _frozen_atomic_numbers: tuple[int, ...]

    supported_properties: ClassVar[tuple[str, ...]] = ("energy", "forces")
    model_identity: ClassVar[str] = GAUSSIAN_CHA_MODEL_IDENTITY

    def __init__(
        self,
        atoms,
        topology: ContinuumChaTopology,
        *,
        expected_topology_sha256: str,
        sigma_e: float,
        order: int = 64,
    ):
        if not isinstance(topology, ContinuumChaTopology):
            raise TypeError("topology must be a ContinuumChaTopology.")
        topology.assert_current(expected_topology_sha256)
        sigma = validate_gaussian_sigma_e(sigma_e)
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError("order must be an integer.")
        if not 8 <= order <= MAX_QUADRATURE_ORDER:
            raise ValueError(f"order must be between 8 and {MAX_QUADRATURE_ORDER}.")
        self._reject_pbc(atoms)
        numbers = _strict_integer_tuple(
            "atomic numbers", atoms.get_atomic_numbers(), count=topology.atom_count
        )
        if numbers != topology.atomic_numbers:
            raise ValueError(
                "Current atomic numbers or atom order differ from topology."
            )
        identity = atoms.arrays.get(_ATOM_IDENTITY_ARRAY)
        if identity is None:
            identity = np.asarray(topology.atom_ids, dtype=np.int64)
            atoms.new_array(_ATOM_IDENTITY_ARRAY, identity)
        identity_values = _strict_integer_tuple(
            "atom identity", identity, count=topology.atom_count
        )
        if identity_values != topology.atom_ids:
            raise ValueError("Current atom identity does not match topology atom IDs.")

        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "expected_topology_sha256", expected_topology_sha256)
        object.__setattr__(self, "sigma_e", sigma)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "_frozen_atom_identity", topology.atom_ids)
        object.__setattr__(self, "_frozen_atomic_numbers", numbers)

    @property
    def provider(self):
        """Expose provider provenance at CalcABC's existing introspection seam."""
        return self

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "provider": "gaussian-cha-continuum",
            "model_identity": self.model_identity,
            "method": "gb",
            "model": "chagb-r6-pbsa-gaussian-sign-v1",
            "profile": "three-site-water-programmatic-experimental-v1",
            "sigma_e": self.sigma_e,
            "quadrature_order": self.order,
            "platform": "CPU",
            "platform_properties": {"dtype": "torch.float64"},
            "placement": "CPU float64 torch",
            "execution": {
                "device": "cpu",
                "dtype": "torch.float64",
                "programmatic_only": True,
                "public_provider_registered": False,
            },
            "derivatives": "torch automatic differentiation of complete scalar",
            "topology_sha256": self.expected_topology_sha256,
            "native_force": True,
            "numerical_force": False,
            "production_admitted": False,
            "accuracy_certified": False,
            "physical_accuracy_claim": False,
            "thermodynamic_quantity": (
                "model-predicted Gaussian-CHA solvation correction"
            ),
        }

    @staticmethod
    def _reject_pbc(atoms) -> None:
        if bool(np.asarray(atoms.get_pbc(), dtype=bool).any()):
            raise ValueError(
                "Gaussian CHA v1 does not support periodic boundary conditions."
            )

    def _validate_atoms(self, atoms) -> None:
        self._reject_pbc(atoms)
        self.topology.assert_current(self.expected_topology_sha256)
        if len(atoms) != self.topology.atom_count:
            raise ValueError("Current atom count differs from the bound topology.")
        numbers = _strict_integer_tuple(
            "atomic numbers",
            atoms.get_atomic_numbers(),
            count=self.topology.atom_count,
        )
        if numbers != self._frozen_atomic_numbers:
            raise ValueError(
                "Current atomic numbers or atom order differ from topology."
            )
        identity = atoms.arrays.get(_ATOM_IDENTITY_ARRAY)
        if identity is None:
            raise ValueError("Current atom identity differs from the bound topology.")
        identity_values = _strict_integer_tuple(
            "atom identity", identity, count=self.topology.atom_count
        )
        if identity_values != self._frozen_atom_identity:
            raise ValueError("Current atom identity differs from the bound topology.")

    def evaluate(
        self, atoms, need_forces: bool = False, calculator=None
    ) -> SolvationResult:
        """Evaluate energy and, when requested, its same-scalar analytic force."""
        import torch

        self._validate_atoms(atoms)
        coordinates = np.array(atoms.get_positions(), dtype=np.float64, copy=True)
        if (
            coordinates.shape != (self.topology.atom_count, 3)
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("positions must be finite and have topology shape [N,3].")
        positions = torch.tensor(
            coordinates,
            dtype=torch.float64,
            device="cpu",
            requires_grad=bool(need_forces),
        )
        scalar = continuum_gaussian_cha_scalar(
            positions,
            self.topology,
            expected_topology_sha256=self.expected_topology_sha256,
            sigma_e=self.sigma_e,
            order=self.order,
        )
        force = None
        if need_forces:
            gradient = torch.autograd.grad(scalar.total_kcal_mol, positions)[0]
            if gradient.shape != positions.shape or not bool(
                torch.isfinite(gradient).all()
            ):
                raise RuntimeError(
                    "Gaussian CHA complete-scalar gradient must be finite [N,3]."
                )
            force = (-gradient / KCAL_PER_MOL_PER_HARTREE).detach().cpu().numpy()
            if force.shape != coordinates.shape or not np.isfinite(force).all():
                raise RuntimeError("Gaussian CHA force must be finite [N,3].")
        total_hartree = (
            float(scalar.total_kcal_mol.detach().cpu()) / KCAL_PER_MOL_PER_HARTREE
        )
        components = {
            "polar": float(scalar.polar_kcal_mol.detach().cpu())
            / KCAL_PER_MOL_PER_HARTREE,
            "cavity": float(scalar.cavity_kcal_mol.detach().cpu())
            / KCAL_PER_MOL_PER_HARTREE,
            "dispersion": float(scalar.dispersion_kcal_mol.detach().cpu())
            / KCAL_PER_MOL_PER_HARTREE,
        }
        point_domain = {
            "scope": scalar.point_domain.scope,
            "active_pairs": scalar.point_domain.active_pairs,
            "fully_occluded_pairs": scalar.point_domain.fully_occluded_pairs,
            "raw_margins": scalar.point_domain.raw_margins,
            "segment_certified": scalar.point_domain.segment_certified,
        }
        diagnostics = {
            "weighted_signs_e": scalar.cha.weighted_signs_e.detach().cpu().tolist(),
            "smoothed_signs": scalar.cha.smoothed_signs.detach().cpu().tolist(),
            "charge_sigma_e": scalar.cha.charge_sigma_e,
            "weighted_signs_over_sigma": scalar.cha.weighted_signs_over_sigma.detach()
            .cpu()
            .tolist(),
            "gaussian_erfc_tails": scalar.cha.gaussian_erfc_tails.detach()
            .cpu()
            .tolist(),
            "gaussian_erfcx_tails": scalar.cha.gaussian_erfcx_tails.detach()
            .cpu()
            .tolist(),
            "electrostatic_size_angstrom": float(
                scalar.cha.electrostatic_size_angstrom.detach().cpu()
            ),
            "born_radii_angstrom": scalar.cha.born_radii_angstrom.detach()
            .cpu()
            .tolist(),
            "cha_factors": scalar.cha.cha_factors.detach().cpu().tolist(),
            "component_sum_residual_hartree": total_hartree - sum(components.values()),
        }
        return SolvationResult(
            energy_hartree=total_hartree,
            forces_hartree_per_angstrom=force,
            components_hartree=components,
            provenance={
                **self.provenance,
                "point_domain": point_domain,
                "polar_diagnostics": diagnostics,
                "radius_provenance": {
                    name: getattr(scalar.radius_provenance, name)
                    for name in (
                        "r6_ses",
                        "polar_algebra",
                        "cavity",
                        "dispersion",
                        "bondi",
                    )
                },
            },
        )

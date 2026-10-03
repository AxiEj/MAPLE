"""Opt-in analytic H/HVP facade for the frozen Gaussian-CHA v2 scalar."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import ClassVar

import numpy as np

from .gaussian_cha_correction import (
    KCAL_PER_MOL_PER_HARTREE,
    GaussianChaCorrection,
)
from .gaussian_cha_profiles import GAUSSIAN_CHA_R6_V2_PROFILE_ID
from .result import SolvationDirectionalResult
from .torch_continuum_chagb import ContinuumChaDomainError
from .torch_continuum_chagb_domain import (
    DomainCertificationFailure,
    certify_three_site_domain,
)
from .torch_continuum_chagb_gaussian import continuum_gaussian_cha_scalar

PINNED_WATER_TOPOLOGY_SHA256 = (
    "2d2659680d850157bf41574257bdae7d593ad166b56eea69dce28c62e54c9391"
)
FROZEN_GAUSSIAN_SIGMA_PANEL_E = (0.001, 0.003, 0.01)
_TRUSTED_ANALYTIC_INTERFACES = (
    "_assert_analytic_identity",
    "preflight_analytic_atoms",
    "get_hessian",
    "get_directional_derivatives",
)
_FROZEN_ANALYTIC_CLASS_METADATA = {
    "mode": "fixed",
    "inner_mode": None,
    "supported_properties": ("energy", "forces", "hessian", "hvp"),
    "model_identity": "chagb-r6-pbsa-gaussian-sign-v1",
}


@dataclass(frozen=True, init=False)
class GaussianChaAnalyticCorrection(GaussianChaCorrection):
    """Strict three-site-water analytic derivative capability.

    Energy and force evaluation remain exactly the inherited implementation.
    Only new H/HVP calls construct a fresh graph from the same frozen scalar.
    """

    supported_properties: ClassVar[tuple[str, ...]] = (
        "energy",
        "forces",
        "hessian",
        "hvp",
    )
    mode: ClassVar[str] = "fixed"
    inner_mode: ClassVar[None] = None
    last_derivative_provenance: dict[str, object] | None
    _derivative_graph_counts: dict[str, int]
    _construction_topology: object
    _construction_interface_methods: tuple[tuple[str, object], ...]
    _construction_class_metadata: tuple[tuple[str, object], ...]

    def __getattribute__(self, name):
        if name in _TRUSTED_ANALYTIC_INTERFACES:
            try:
                frozen = object.__getattribute__(
                    self, "_construction_interface_methods"
                )
            except AttributeError:
                frozen = ()
            for frozen_name, descriptor in frozen:
                if frozen_name == name:
                    return descriptor.__get__(self, type(self))
        return super().__getattribute__(name)

    def __init__(
        self,
        atoms,
        topology,
        *,
        expected_topology_sha256: str,
        sigma_e: float,
        order: int = 64,
        numerical_profile_id: str = GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    ):
        if expected_topology_sha256 != PINNED_WATER_TOPOLOGY_SHA256 or (
            getattr(topology, "content_sha256", None) != PINNED_WATER_TOPOLOGY_SHA256
        ):
            raise ValueError(
                "Gaussian-CHA analytic derivatives require the pinned water topology."
            )
        if sigma_e not in FROZEN_GAUSSIAN_SIGMA_PANEL_E:
            raise ValueError(
                "Gaussian-CHA analytic sigma must belong to the frozen three-width panel."
            )
        if order != 64:
            raise ValueError(
                "Gaussian-CHA analytic derivatives require production order 64."
            )
        if numerical_profile_id != GAUSSIAN_CHA_R6_V2_PROFILE_ID:
            raise ValueError(
                "Gaussian-CHA analytic derivatives require the exact v2 numerical profile."
            )
        super().__init__(
            atoms,
            topology,
            expected_topology_sha256=expected_topology_sha256,
            sigma_e=sigma_e,
            order=order,
            numerical_profile_id=numerical_profile_id,
        )
        object.__setattr__(self, "_construction_topology", self.topology)
        object.__setattr__(
            self,
            "_construction_interface_methods",
            tuple(
                (name, inspect.getattr_static(type(self), name))
                for name in _TRUSTED_ANALYTIC_INTERFACES
            ),
        )
        object.__setattr__(
            self,
            "_construction_class_metadata",
            tuple(
                (name, getattr(type(self), name))
                for name in _FROZEN_ANALYTIC_CLASS_METADATA
            ),
        )
        object.__setattr__(self, "last_derivative_provenance", None)
        object.__setattr__(
            self,
            "_derivative_graph_counts",
            {"dense-hessian": 0, "directional-hvp": 0},
        )
        self._assert_analytic_identity()

    def _assert_analytic_identity(self) -> None:
        instance_values = object.__getattribute__(self, "__dict__")
        for name, descriptor in self._construction_interface_methods:
            if name in instance_values:
                raise ValueError(
                    f"Gaussian-CHA analytic reviewed interface {name!r} was shadowed."
                )
            if inspect.getattr_static(type(self), name) is not descriptor:
                raise ValueError(
                    f"Gaussian-CHA analytic reviewed interface {name!r} drifted."
                )
        for name, expected in self._construction_class_metadata:
            if name in instance_values:
                raise ValueError(
                    f"Gaussian-CHA analytic frozen metadata {name!r} was shadowed."
                )
            if (
                expected != _FROZEN_ANALYTIC_CLASS_METADATA[name]
                or getattr(type(self), name) != expected
            ):
                raise ValueError(
                    f"Gaussian-CHA analytic frozen metadata {name!r} drifted."
                )
        if self.topology is not self._construction_topology:
            raise ValueError("Gaussian-CHA analytic topology object was replaced.")
        try:
            self.topology.assert_current(PINNED_WATER_TOPOLOGY_SHA256)
        except Exception as exc:
            raise ValueError(
                "Gaussian-CHA analytic identity topology drifted."
            ) from exc
        if (
            self.expected_topology_sha256 != PINNED_WATER_TOPOLOGY_SHA256
            or self.topology.content_sha256 != PINNED_WATER_TOPOLOGY_SHA256
            or self.sigma_e not in FROZEN_GAUSSIAN_SIGMA_PANEL_E
            or self.order != 64
            or self.numerical_profile_id != GAUSSIAN_CHA_R6_V2_PROFILE_ID
        ):
            raise ValueError("Gaussian-CHA analytic identity fingerprint drifted.")

    @property
    def analytic_task_derivatives_admitted(self) -> bool:
        """Report narrow runtime API compatibility, not scientific qualification."""
        try:
            self._assert_analytic_identity()
        except (TypeError, ValueError):
            return False
        return True

    @property
    def analytic_identity_fingerprint(self) -> tuple[object, ...]:
        self._assert_analytic_identity()
        return (
            self.model_identity,
            self.numerical_profile_id,
            self.topology.content_sha256,
            self.sigma_e,
            self.order,
            "cpu",
            "torch.float64",
        )

    @property
    def analytic_identity_provenance(self) -> dict[str, object]:
        self._assert_analytic_identity()
        return {
            "model_identity": self.model_identity,
            "numerical_profile_id": self.numerical_profile_id,
            "topology_sha256": self.topology.content_sha256,
            "sigma_e": self.sigma_e,
            "order": self.order,
            "device": "cpu",
            "dtype": "torch.float64",
        }

    @property
    def derivative_graph_counts(self) -> dict[str, int]:
        return dict(self._derivative_graph_counts)

    @property
    def analytic_graph_counters(self) -> dict[str, int]:
        counts = self.derivative_graph_counts
        return {
            "dense_hessian": counts["dense-hessian"],
            "directional_hvp": counts["directional-hvp"],
            "total": sum(counts.values()),
        }

    def preflight_analytic_atoms(self, atoms) -> np.ndarray:
        """Check cheap bound identity and the existing three-site point domain.

        Contact-node conditioning remains part of the subsequently requested
        solvent graph; this preflight deliberately does not duplicate it or the
        complete scalar.
        """
        import torch

        self._assert_analytic_identity()
        self._validate_atoms(atoms)
        coordinates = np.asarray(atoms.get_positions(), dtype=np.float64)
        if coordinates.shape != (3, 3) or not np.isfinite(coordinates).all():
            raise ValueError("Gaussian-CHA analytic positions must be finite [3,3].")
        positions = torch.tensor(
            np.array(coordinates, copy=True), dtype=torch.float64, device="cpu"
        )
        fixed = self.topology.tensors(
            expected_content_sha256=self.expected_topology_sha256,
            device="cpu",
        )
        point_domain = certify_three_site_domain(positions, fixed.cha_radii_angstrom)
        if isinstance(point_domain, DomainCertificationFailure):
            raise ContinuumChaDomainError(point_domain)
        return np.array(coordinates, dtype=np.float64, copy=True)

    def _scalar_leaf_hartree(self, atoms):
        import torch

        coordinates = self.preflight_analytic_atoms(atoms)
        positions = torch.tensor(
            np.array(coordinates, copy=True),
            dtype=torch.float64,
            device="cpu",
            requires_grad=True,
        )
        scalar = continuum_gaussian_cha_scalar(
            positions,
            self.topology,
            expected_topology_sha256=self.expected_topology_sha256,
            sigma_e=self.sigma_e,
            order=self.order,
            numerical_profile_id=self.numerical_profile_id,
        )
        energy_hartree = scalar.total_kcal_mol / KCAL_PER_MOL_PER_HARTREE
        if energy_hartree.shape != () or not bool(torch.isfinite(energy_hartree)):
            raise RuntimeError("Gaussian-CHA analytic scalar must be finite.")
        return positions, energy_hartree, scalar

    def _record_derivative(self, derivative: str, scalar) -> dict[str, object]:
        counts = self._derivative_graph_counts
        counts[derivative] += 1
        provenance = {
            **self.provenance,
            **self.analytic_identity_provenance,
            "provider": "gaussian-cha-analytic-continuum",
            "derivative": derivative,
            "runtime_finite_difference": False,
            "analytic_interface_status": "experimental-programmatic-api",
            "scientific_qualification_receipt": False,
            "graph_construction_count": counts[derivative],
            "components_hartree": {
                "polar": float(
                    scalar.polar_kcal_mol.detach().cpu() / KCAL_PER_MOL_PER_HARTREE
                ),
                "cavity": float(
                    scalar.cavity_kcal_mol.detach().cpu() / KCAL_PER_MOL_PER_HARTREE
                ),
                "dispersion": float(
                    scalar.dispersion_kcal_mol.detach().cpu() / KCAL_PER_MOL_PER_HARTREE
                ),
            },
        }
        object.__setattr__(self, "last_derivative_provenance", provenance)
        return provenance

    def get_hessian(self, atoms) -> np.ndarray:
        """Materialize only the requested solvent Hessian in Hartree/A^2."""
        import torch

        positions, energy, scalar = self._scalar_leaf_hartree(atoms)
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0].reshape(
            -1
        )
        rows = [
            torch.autograd.grad(value, positions, retain_graph=True)[0].reshape(-1)
            for value in gradient
        ]
        hessian = torch.stack(rows).detach().cpu().numpy()
        if hessian.shape != (9, 9) or not np.isfinite(hessian).all():
            raise RuntimeError("Gaussian-CHA analytic Hessian must be finite [9,9].")
        self._record_derivative("dense-hessian", scalar)
        return hessian

    @staticmethod
    def _strict_direction(direction) -> np.ndarray:
        if not isinstance(direction, np.ndarray) or direction.dtype != np.float64:
            raise TypeError(
                "Gaussian-CHA HVP direction must be an explicit float64 array."
            )
        if direction.shape != (9,) or not np.isfinite(direction).all():
            raise ValueError(
                "Gaussian-CHA HVP direction must be finite flat shape (9,)."
            )
        return np.array(direction, dtype=np.float64, copy=True)

    def get_directional_derivatives(
        self, atoms, direction
    ) -> SolvationDirectionalResult:
        """Return solvent E/F/Hv from one graph without forming dense H."""
        import torch

        values = self._strict_direction(direction)
        positions, energy, scalar = self._scalar_leaf_hartree(atoms)
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
        vector = torch.as_tensor(
            values.reshape(positions.shape), dtype=torch.float64, device="cpu"
        )
        product = torch.autograd.grad(
            gradient,
            positions,
            grad_outputs=vector,
        )[0]
        forces = -gradient.detach().cpu().numpy()
        hvp = product.reshape(-1).detach().cpu().numpy()
        if (
            forces.shape != (3, 3)
            or hvp.shape != (9,)
            or not np.isfinite(forces).all()
            or not np.isfinite(hvp).all()
        ):
            raise RuntimeError("Gaussian-CHA directional derivatives must be finite.")
        provenance = self._record_derivative("directional-hvp", scalar)
        return SolvationDirectionalResult(
            hvp_hartree_per_angstrom2=hvp,
            forces_hartree_per_angstrom=forces,
            energy_hartree=float(energy.detach().cpu()),
            provenance=provenance,
        )

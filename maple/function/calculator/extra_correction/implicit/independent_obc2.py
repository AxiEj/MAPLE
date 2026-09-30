"""OpenMM-independent Torch OBC-II/mbondi2 solvent provider."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from maple.function.calculator.analytic_protocol import (
    ScalarCellIdentity,
    ScalarPointIdentity,
    freeze_identity_options,
    geometry_sha256,
)

from .obc2_parameters import build_obc2_parameters
from .obc2_profile import OBC2_FORMULA_MANIFEST_ID
from .obc2_topology import OBC2Topology
from .radii import IndependentMbondi2RadiusProvider
from .result import SolvationDirectionalResult, SolvationResult
from .torch_dense_budget import DEFAULT_DENSE_TORCH_BUDGET, DenseTorchBudget
from .torch_obc2 import KINK_MARGIN_NM, TorchOBC2


class IndependentOBC2:
    """Exact independent OBC-II/ACE-or-none E/F/H/HVP implementation."""

    supported_properties = frozenset({"energy", "forces", "hessian", "hvp"})

    def __init__(
        self,
        atoms,
        charges,
        *,
        nonpolar: str = "ace",
        device="cpu",
        dtype=None,
        resource_budget: DenseTorchBudget = DEFAULT_DENSE_TORCH_BUDGET,
    ) -> None:
        if not isinstance(nonpolar, str):
            raise TypeError("Independent OBC-II nonpolar must be an explicit string.")
        nonpolar = nonpolar.lower()
        if nonpolar not in {"ace", "none"}:
            raise ValueError("Independent OBC-II nonpolar must be 'ace' or 'none'.")
        self.topology = OBC2Topology.from_atoms(atoms)
        self.charges = np.asarray(charges, dtype=np.float64).copy()
        if self.charges.shape != (len(self.topology.atoms),):
            raise ValueError("Independent OBC-II requires one charge per topology atom.")
        if not np.isfinite(self.charges).all():
            raise ValueError("Independent OBC-II charges must be finite.")
        self.charges.setflags(write=False)
        self.radius_provider = IndependentMbondi2RadiusProvider()
        self.radius_result = self.radius_provider.assign(self.topology)
        self.parameters = build_obc2_parameters(
            self.charges,
            self.radius_result,
            nonpolar=nonpolar,
        )
        self.kernel = TorchOBC2(
            self.parameters,
            device=device,
            dtype=dtype,
            resource_budget=resource_budget,
        )
        self.model = "obc2"
        self.profile = "obc2-mbondi2"
        self.nonpolar = nonpolar
        self.device = self.kernel.device
        self.dtype = self.kernel.dtype
        self.platform = "torch"
        self.last_derivative_provenance: dict[str, Any] | None = None
        self._cell_identity = ScalarCellIdentity(
            component="solvent",
            implementation="torch-obc2-v2",
            model="obc2-mbondi2",
            checkpoint_path=None,
            checkpoint_sha256=None,
            device=str(self.device),
            effective_dtype=str(self.dtype),
            d4=False,
            source_energy_unit="hartree",
            cutoff_guard_policy=None,
            cutoff_guard_margin_angstrom=None,
            options=freeze_identity_options(
                {
                    "canonical_payload_sha256": self.parameters.provenance[
                        "canonical_payload_sha256"
                    ],
                    "equation_profile": self.parameters.provenance[
                        "equation_profile"
                    ],
                    "formula_manifest_id": OBC2_FORMULA_MANIFEST_ID,
                    "formula_source_sha256": self.parameters.provenance[
                        "formula_source_sha256"
                    ],
                    "nonpolar": nonpolar,
                    "offset_nm": float(self.parameters.provenance["offset_nm"]),
                    "ordered_charge_sha256": self.parameters.provenance[
                        "ordered_charge_sha256"
                    ],
                    "preparation_provider": "maple-independent-mbondi2",
                    "radius_profile": "obc2-mbondi2",
                    "runtime_provider": "torch-obc2",
                    "solute_dielectric": float(self.parameters.solute_dielectric),
                    "solvent_dielectric": float(self.parameters.solvent_dielectric),
                }
            ),
        )
        budget = asdict(resource_budget)
        self._provenance = {
            **self.kernel.provenance,
            "provider": "torch-obc2",
            "runtime_provider": "torch-obc2",
            "preparation_provider": "maple-independent-mbondi2",
            "oracle_provider": None,
            "implementation_profile": "torch-obc2-v2",
            "method": "gb",
            "model": "obc2",
            "amber_igb": 5,
            "profile": "obc2-mbondi2",
            "nonpolar": nonpolar,
            "formula_manifest_id": OBC2_FORMULA_MANIFEST_ID,
            "topology_sha256": self.topology.identity_sha256,
            "mol2_identity_sha256": self.topology.mol2_identity_sha256,
            "ordered_charge_sha256": self.parameters.provenance[
                "ordered_charge_sha256"
            ],
            "dtype": str(self.dtype),
            "device": str(self.device),
            "platform": "torch",
            "supported_properties": sorted(self.supported_properties),
            "resource_budget": budget,
            "branch_rejection_margin_nm": KINK_MARGIN_NM,
            "runtime_openmm_required": False,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            **self._provenance,
            "radius_provider": dict(self.radius_result.provenance),
        }

    @property
    def analytic_cell_identity(self) -> ScalarCellIdentity:
        return self._cell_identity

    def analytic_scalar_identity(self, atoms) -> ScalarPointIdentity:
        self._validate_bound_atoms(atoms)
        return ScalarPointIdentity(
            cell=self._cell_identity,
            geometry_sha256=geometry_sha256(atoms),
            atomic_numbers=tuple(int(value) for value in atoms.get_atomic_numbers()),
            charge=float(atoms.info.get("charge", 0.0)),
            multiplicity=int(atoms.info.get("mult", 1)),
            atom_topology_sha256=self.topology.identity_sha256,
            neighbor_graph_sha256=None,
            active_cutoffs_angstrom=(),
            minimum_cutoff_gap_angstrom=None,
            response_settings=freeze_identity_options(
                {
                    "branch_rejection_margin_nm": KINK_MARGIN_NM,
                    "fixed_charges": True,
                    "salt_kappa_inverse_nm": 0.0,
                }
            ),
        )

    def derivative_backend(self) -> "IndependentOBC2":
        return self

    def _validate_bound_atoms(self, atoms) -> None:
        current = OBC2Topology.from_atoms(atoms)
        if current.identity_sha256 != self.topology.identity_sha256:
            raise ValueError(
                "Independent OBC-II is bound to the frozen atom order and MOL2 topology."
            )

    def branch_diagnostics(self, atoms) -> dict[str, float]:
        self._validate_bound_atoms(atoms)
        return self.kernel.branch_diagnostics(atoms)

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        self._validate_bound_atoms(atoms)
        result = self.kernel.evaluate(
            atoms,
            need_forces=need_forces,
            calculator=calculator,
        )
        return SolvationResult(
            energy_hartree=result.energy_hartree,
            forces_hartree_per_angstrom=result.forces_hartree_per_angstrom,
            components_hartree=dict(result.components_hartree),
            provenance={**self.provenance, **dict(result.provenance)},
        )

    def hessian(self, atoms) -> np.ndarray:
        self._validate_bound_atoms(atoms)
        value = self.kernel.hessian(atoms)
        self.last_derivative_provenance = dict(
            self.kernel.last_derivative_provenance or {}
        )
        return value

    def directional_derivatives(
        self, atoms, direction
    ) -> SolvationDirectionalResult:
        self._validate_bound_atoms(atoms)
        result = self.kernel.directional_derivatives(atoms, direction)
        provenance = {**self.provenance, **dict(result.provenance)}
        self.last_derivative_provenance = provenance
        return SolvationDirectionalResult(
            hvp_hartree_per_angstrom2=result.hvp_hartree_per_angstrom2,
            forces_hartree_per_angstrom=result.forces_hartree_per_angstrom,
            energy_hartree=result.energy_hartree,
            provenance=provenance,
        )

    def hvp(self, atoms, direction) -> np.ndarray:
        return self.directional_derivatives(
            atoms, direction
        ).hvp_hartree_per_angstrom2


"""Fail-closed research adapter for the supplied MACE-POLAR-EF TorchScript.

The electronic source used for continuum coupling is defined by the derivative
of the checkpoint energy with respect to the atomwise external potential and
potential gradient.  The checkpoint's auxiliary ``density_coefficients`` output
is retained as a diagnostic only; it is not assumed to be energy-conjugate.

This module is intentionally not registered as a public MAPLE calculator.  It
supports first derivatives on non-periodic molecular graphs, binds the exact
checkpoint bytes and traced wrapper semantics, and exposes an independent
uniform-field concavity audit.  Passing autograd consistency alone is not an
electronic passivity proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import zipfile

import numpy as np
import torch

from .electrostatic_pairing import MACE_POLAR_L1_PAIRING

MACE_POLAR_EF_MODEL_ID = "mace-polar-ef-v2-energy-functional"
MACE_POLAR_EF_CHECKPOINT_SHA256 = (
    "4f820d381d06bbb37b02574c38da7203e5d429407fa2a512231fc08cdbb69b6b"
)
MACE_POLAR_EF_CHECKPOINT_SIZE = 34_581_570
MACE_POLAR_EF_AUTOGRAD_ORDER = 1
MACE_POLAR_EF_PASSIVITY_FIELD_STEP = 1.0e-3
MACE_POLAR_EF_MAXIMUM_POSITIVE_CURVATURE = 1.0e-2


def _wrapper_source(checkpoint_bytes: bytes) -> tuple[str, str]:
    buffer = io.BytesIO(checkpoint_bytes)
    if not zipfile.is_zipfile(buffer):
        raise ValueError("MACE-POLAR-EF checkpoint must be a TorchScript zip archive.")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        candidates = [
            name for name in archive.namelist() if name.endswith("/code/__torch__.py")
        ]
        if len(candidates) != 1:
            raise ValueError("Unable to identify the checkpoint wrapper source.")
        source = archive.read(candidates[0]).decode("utf-8")
    required = (
        "class PolarMACEEFTraceable",
        "external_field0 = torch.neg(external_field)",
        "external_potential_values = torch.unsqueeze(external_potential, -1)",
    )
    if any(text not in source for text in required):
        raise ValueError("Checkpoint wrapper does not expose the audited EF semantics.")
    return source, hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MACEPolarEFConfig:
    """Exact runtime binding for one molecular MACE-POLAR-EF evaluation."""

    checkpoint_path: str
    atomic_numbers: tuple[int, ...]
    total_charge: int = 0
    spin_multiplicity: int = 1
    device: str = "cuda"
    cutoff_margin_angstrom: float = 0.05
    _checkpoint_bytes: bytes = field(init=False, repr=False, compare=False)
    _wrapper_source_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_bytes = path.read_bytes()
        if len(checkpoint_bytes) != MACE_POLAR_EF_CHECKPOINT_SIZE:
            raise ValueError("MACE-POLAR-EF checkpoint size does not match v2.")
        if (
            hashlib.sha256(checkpoint_bytes).hexdigest()
            != MACE_POLAR_EF_CHECKPOINT_SHA256
        ):
            raise ValueError("MACE-POLAR-EF checkpoint SHA256 does not match v2.")
        _source, wrapper_sha256 = _wrapper_source(checkpoint_bytes)
        object.__setattr__(self, "checkpoint_path", str(path))
        object.__setattr__(self, "_checkpoint_bytes", checkpoint_bytes)
        object.__setattr__(self, "_wrapper_source_sha256", wrapper_sha256)

        numbers = tuple(self.atomic_numbers)
        if not numbers or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 83
            for value in numbers
        ):
            raise ValueError("atomic_numbers must contain integers in [1, 83].")
        object.__setattr__(self, "atomic_numbers", numbers)

        for name in ("total_charge", "spin_multiplicity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.spin_multiplicity < 1:
            raise ValueError("spin_multiplicity must be at least one.")
        electron_count = sum(numbers) - self.total_charge
        if (
            electron_count <= 0
            or electron_count % 2 != (self.spin_multiplicity - 1) % 2
        ):
            raise ValueError(
                "MACE-POLAR-EF charge and spin multiplicity violate electron-count parity."
            )
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("The supplied traced checkpoint is CUDA-only.")
        if device.index is None:
            device = torch.device("cuda", 0)
        if device.index != 0:
            raise ValueError(
                "The supplied traced checkpoint embeds CUDA device 0 constants."
            )
        object.__setattr__(self, "device", str(device))
        margin = float(self.cutoff_margin_angstrom)
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError("cutoff_margin_angstrom must be finite and positive.")
        object.__setattr__(self, "cutoff_margin_angstrom", margin)

    @property
    def atom_count(self) -> int:
        return len(self.atomic_numbers)

    @property
    def wrapper_source_sha256(self) -> str:
        return self._wrapper_source_sha256

    def as_provenance(self) -> dict[str, object]:
        return {
            **self.as_identity(),
            "checkpoint_path": self.checkpoint_path,
        }

    def as_identity(self) -> dict[str, object]:
        """Return path-independent scientific and inference identity."""

        return {
            "model_id": MACE_POLAR_EF_MODEL_ID,
            "checkpoint_sha256": MACE_POLAR_EF_CHECKPOINT_SHA256,
            "checkpoint_size": MACE_POLAR_EF_CHECKPOINT_SIZE,
            "wrapper_source_sha256": self.wrapper_source_sha256,
            "atomic_numbers": list(self.atomic_numbers),
            "total_charge": self.total_charge,
            "spin_multiplicity": self.spin_multiplicity,
            "checkpoint_spin_input_semantics": "multiplicity (singlet=1)",
            "device": self.device,
            "dtype": "torch.float32",
            "field_input": "atomwise [potential, grad_x, grad_y, grad_z]",
            "maple_vector_input": "electrostatic potential gradient grad(V)",
            "checkpoint_vector_input": "physical electric field E=-grad(V)",
            "wrapper_internal_vector": "grad(V) after its built-in negation",
            "external_potential_role": "explicit q_i*V_i energy coupling",
            "external_gradient_role": "learned field response and energy coupling",
            "source_definition": "Q * dE_model/d[potential,gradient]",
            "density_coefficients_role": "diagnostic-only",
            "checkpoint_converter_lineage": None,
            "checkpoint_upstream_provenance_available": False,
            "passivity_gate_required_before_continuum_coupling": True,
            "autograd_order": MACE_POLAR_EF_AUTOGRAD_ORDER,
            "cutoff_margin_angstrom": self.cutoff_margin_angstrom,
            "publicly_registered": False,
            "release_admitted": False,
        }


@dataclass(frozen=True, slots=True)
class MACEPolarEFEvaluation:
    energy_ev: float
    conjugate_source_raw: np.ndarray
    coordinate_gradient_ev_per_angstrom: np.ndarray
    density_coefficients_diagnostic: np.ndarray
    field_cartesian: np.ndarray

    def __post_init__(self) -> None:
        energy = float(self.energy_ev)
        if not np.isfinite(energy):
            raise ValueError("MACE-POLAR-EF energy must be finite.")
        object.__setattr__(self, "energy_ev", energy)
        expected_atoms: int | None = None
        for name, columns in (
            ("conjugate_source_raw", 4),
            ("coordinate_gradient_ev_per_angstrom", 3),
            ("density_coefficients_diagnostic", 4),
            ("field_cartesian", 4),
        ):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                values.ndim != 2
                or values.shape[1] != columns
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f"{name} must be finite with shape (n_atoms, {columns})."
                )
            if expected_atoms is None:
                expected_atoms = values.shape[0]
            elif values.shape[0] != expected_atoms:
                raise ValueError("MACE-POLAR-EF result atom counts must match.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class MACEPolarEFResponseAudit:
    """Uniform affine-potential concavity evidence for one fixed geometry."""

    field_step_ev_per_e_angstrom: float
    energy_hessian: np.ndarray
    energy_hessian_eigenvalues: np.ndarray
    maximum_positive_curvature: float
    tolerance: float
    passivity_passed: bool
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        hessian = np.array(self.energy_hessian, dtype=float, copy=True)
        eigenvalues = np.array(self.energy_hessian_eigenvalues, dtype=float, copy=True)
        if (
            hessian.shape != (3, 3)
            or eigenvalues.shape != (3,)
            or not (np.all(np.isfinite(hessian)) and np.all(np.isfinite(eigenvalues)))
        ):
            raise ValueError("MACE-POLAR-EF passivity evidence must be finite 3D data.")
        hessian.setflags(write=False)
        eigenvalues.setflags(write=False)
        object.__setattr__(self, "energy_hessian", hessian)
        object.__setattr__(self, "energy_hessian_eigenvalues", eigenvalues)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class MACEPolarEFEnergyModel:
    """Exact-checkpoint molecular evaluator with an energy-conjugate source."""

    def __init__(self, config: MACEPolarEFConfig) -> None:
        if not isinstance(config, MACEPolarEFConfig):
            raise TypeError("config must be MACEPolarEFConfig.")
        if not torch.cuda.is_available():
            raise RuntimeError("MACE-POLAR-EF requires an available CUDA device.")
        self.config = config
        self.device = torch.device(config.device)
        self.model = torch.jit.load(
            io.BytesIO(config._checkpoint_bytes), map_location=self.device
        ).eval()
        schema = str(self.model.forward.schema)
        expected_schema_parts = (
            "Tensor external_field",
            "Tensor external_potential",
            "Tensor local_or_ghost",
            "-> ((Tensor, Tensor, Tensor))",
        )
        if any(part not in schema for part in expected_schema_parts):
            raise ValueError(
                "MACE-POLAR-EF forward schema is not the audited v2 schema."
            )
        model_numbers = tuple(
            int(value) for value in self.model.atomic_numbers.tolist()
        )
        if model_numbers != tuple(range(1, 84)):
            raise ValueError("MACE-POLAR-EF element table does not match v2.")
        self.cutoff_angstrom = float(self.model.r_max.detach().cpu())
        if not np.isclose(self.cutoff_angstrom, 6.0, atol=0.0, rtol=0.0):
            raise ValueError("MACE-POLAR-EF cutoff does not match v2.")
        self.runtime_provenance = MappingProxyType(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "device": str(self.device),
                "device_name": torch.cuda.get_device_name(self.device),
                "checkpoint_loader": "torch.jit.load(verified-in-memory-bytes)",
            }
        )

    @property
    def atom_count(self) -> int:
        return self.config.atom_count

    def _validate_tensors(
        self,
        positions: torch.Tensor,
        field_cartesian: torch.Tensor,
    ) -> None:
        if positions.dtype != torch.float32 or field_cartesian.dtype != torch.float32:
            raise TypeError("MACE-POLAR-EF tensors must use torch.float32.")
        if (
            positions.device != self.device
            or field_cartesian.device != positions.device
        ):
            raise ValueError("MACE-POLAR-EF tensors must share one CUDA device.")
        if positions.shape != (self.atom_count, 3):
            raise ValueError(f"positions must have shape ({self.atom_count}, 3).")
        if field_cartesian.shape != (self.atom_count, 4):
            raise ValueError(f"field_cartesian must have shape ({self.atom_count}, 4).")
        if not bool(torch.isfinite(positions).all()) or not bool(
            torch.isfinite(field_cartesian).all()
        ):
            raise ValueError("MACE-POLAR-EF tensors must be finite.")

    def _graph(self, positions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        distances = torch.cdist(positions.detach(), positions.detach())
        diagonal = torch.eye(self.atom_count, dtype=torch.bool, device=positions.device)
        off_diagonal = ~diagonal
        near_cutoff = torch.abs(distances - self.cutoff_angstrom) < (
            self.config.cutoff_margin_angstrom
        )
        if bool((near_cutoff & off_diagonal).any()):
            raise ValueError("A pair lies inside the MACE cutoff topology margin.")
        edge_mask = (distances < self.cutoff_angstrom) & off_diagonal
        edge_index = torch.nonzero(edge_mask, as_tuple=False).T.contiguous()
        if edge_index.shape[1] == 0 and self.atom_count > 1:
            raise ValueError("MACE-POLAR-EF molecular graph contains no edges.")

        node_attributes = positions.new_zeros((self.atom_count, 83))
        for atom_index, atomic_number in enumerate(self.config.atomic_numbers):
            node_attributes[atom_index, atomic_number - 1] = 1.0
        edge_count = edge_index.shape[1]
        shifts = positions.new_zeros((edge_count, 3))
        unit_shifts = positions.new_zeros((edge_count, 3))
        batch = torch.zeros(self.atom_count, dtype=torch.long, device=positions.device)
        ptr = torch.tensor(
            [0, self.atom_count], dtype=torch.long, device=positions.device
        )
        cell = positions.new_zeros((3, 3))
        total_charge = positions.new_tensor([float(self.config.total_charge)])
        total_spin = positions.new_tensor([float(self.config.spin_multiplicity)])
        local_or_ghost = positions.new_ones((self.atom_count,))
        return (
            node_attributes,
            edge_index,
            shifts,
            unit_shifts,
            batch,
            ptr,
            cell,
            total_charge,
            total_spin,
            local_or_ghost,
        )

    def energy_torch(
        self,
        positions: torch.Tensor,
        field_cartesian: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total energy, node energies, and diagnostic density output."""

        self._validate_tensors(positions, field_cartesian)
        graph = self._graph(positions)
        (
            node_attributes,
            edge_index,
            shifts,
            unit_shifts,
            batch,
            ptr,
            cell,
            total_charge,
            total_spin,
            local_or_ghost,
        ) = graph
        # MAPLE/pyddx carries the electrostatic potential jet [V, grad(V)].
        # The traced wrapper API is field-facing and negates its vector input
        # before the underlying PolarMACE energy, so provide E=-grad(V) here.
        checkpoint_electric_field = -field_cartesian[:, 1:]
        checkpoint_potential = field_cartesian[:, 0]
        energy, node_energy, density = self.model(
            positions,
            node_attributes,
            edge_index,
            shifts,
            unit_shifts,
            batch,
            ptr,
            cell,
            total_charge,
            total_spin,
            checkpoint_electric_field,
            checkpoint_potential,
            local_or_ghost,
        )
        if energy.shape != (1,) or node_energy.shape != (self.atom_count,):
            raise RuntimeError("MACE-POLAR-EF energy outputs have unexpected shapes.")
        if density.shape != (self.atom_count, 4):
            raise RuntimeError("MACE-POLAR-EF density output has an unexpected shape.")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (energy, node_energy, density)
        ):
            raise FloatingPointError("MACE-POLAR-EF returned a non-finite tensor.")
        return energy.reshape(()), node_energy, density

    @staticmethod
    def _raw_source_from_cartesian_gradient(
        field_gradient_cartesian: torch.Tensor,
    ) -> torch.Tensor:
        indices = torch.tensor(
            MACE_POLAR_L1_PAIRING.density_to_field_indices,
            dtype=torch.long,
            device=field_gradient_cartesian.device,
        )
        inverse = torch.argsort(indices)
        return torch.index_select(field_gradient_cartesian, -1, inverse)

    def conjugate_source_torch(
        self,
        positions: torch.Tensor,
        field_cartesian: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return energy and ``Q*dE/df`` in MAPLE raw source order."""

        if create_graph:
            raise NotImplementedError(
                "MACE-POLAR-EF exposes first derivatives only; second-order "
                "autograd is an audit-only operation implemented separately."
            )
        if not field_cartesian.requires_grad:
            raise ValueError("field_cartesian must require gradients.")
        energy, _node_energy, density = self.energy_torch(positions, field_cartesian)
        (field_gradient,) = torch.autograd.grad(
            energy,
            field_cartesian,
            create_graph=create_graph,
            retain_graph=True,
        )
        if not bool(torch.isfinite(field_gradient).all()):
            raise FloatingPointError("MACE-POLAR-EF field gradient is non-finite.")
        source_raw = self._raw_source_from_cartesian_gradient(field_gradient)
        return energy, source_raw, density

    @staticmethod
    def _cartesian_gradient_from_raw_source(source_raw: np.ndarray) -> np.ndarray:
        return MACE_POLAR_L1_PAIRING.density_to_field_order(source_raw)

    def _affine_field_energy_gradient(
        self,
        positions: torch.Tensor,
        potential_gradient: np.ndarray,
    ) -> np.ndarray:
        gradient = np.asarray(potential_gradient, dtype=np.float32)
        if gradient.shape != (3,) or not np.all(np.isfinite(gradient)):
            raise ValueError("potential_gradient must be one finite Cartesian vector.")
        centered = np.asarray(positions.detach().cpu(), dtype=np.float32)
        centered = centered - np.mean(centered, axis=0, keepdims=True)
        field_values = np.zeros((self.atom_count, 4), dtype=np.float32)
        field_values[:, 0] = centered @ gradient
        field_values[:, 1:] = gradient
        field = torch.tensor(
            field_values,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        _energy, source_raw, _density = self.conjugate_source_torch(
            positions,
            field,
        )
        cartesian = self._cartesian_gradient_from_raw_source(
            np.asarray(source_raw.detach().cpu(), dtype=float)
        )
        return np.sum(
            cartesian[:, 0, None] * centered + cartesian[:, 1:],
            axis=0,
        )

    def audit_uniform_field_passivity(
        self,
        positions_angstrom: np.ndarray,
        *,
        field_step_ev_per_e_angstrom: float = MACE_POLAR_EF_PASSIVITY_FIELD_STEP,
        maximum_positive_curvature: float = MACE_POLAR_EF_MAXIMUM_POSITIVE_CURVATURE,
    ) -> MACEPolarEFResponseAudit:
        """Test concavity along coherent uniform-field potentials.

        A physical ground-state energy is concave in an applied electrostatic
        potential, so the uniform-field energy Hessian must be negative
        semidefinite (polarizability is its negative).  Both the atomwise
        potential values and their common gradient are supplied; testing a
        gradient without its affine potential would not represent one coherent
        electrostatic field.
        """

        step = float(field_step_ev_per_e_angstrom)
        tolerance = float(maximum_positive_curvature)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("field_step_ev_per_e_angstrom must be positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("maximum_positive_curvature must be nonnegative.")
        positions = torch.tensor(
            np.asarray(positions_angstrom, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        zero_field = torch.zeros(
            (self.atom_count, 4), dtype=torch.float32, device=self.device
        )
        self._validate_tensors(positions, zero_field)
        hessian = np.empty((3, 3), dtype=float)
        for axis in range(3):
            direction = np.zeros(3, dtype=np.float32)
            direction[axis] = step
            plus = self._affine_field_energy_gradient(positions, direction)
            minus = self._affine_field_energy_gradient(positions, -direction)
            hessian[:, axis] = (plus - minus) / (2.0 * step)
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues = np.linalg.eigvalsh(hessian)
        maximum = float(np.max(eigenvalues))
        passed = maximum <= tolerance
        return MACEPolarEFResponseAudit(
            field_step_ev_per_e_angstrom=step,
            energy_hessian=hessian,
            energy_hessian_eigenvalues=eigenvalues,
            maximum_positive_curvature=maximum,
            tolerance=tolerance,
            passivity_passed=passed,
            provenance={
                "criterion": "ground-state-energy-concavity-in-affine-external-potential",
                "polarizability_definition": "alpha=-d2E/dE2",
                "coordinate_origin": "unweighted-atomic-centroid",
                "external_drive": "per-atom V_i=g dot (R_i-centroid), grad(V)_i=g",
                "energy_hessian_units": "e^2 angstrom^2/eV",
                "positive_curvature_tolerance_rationale": (
                    "100x float32 source-difference noise allowance"
                ),
                "checkpoint_sha256": MACE_POLAR_EF_CHECKPOINT_SHA256,
                "spin_multiplicity": self.config.spin_multiplicity,
                "passed": passed,
            },
        )

    def evaluate(
        self,
        positions_angstrom: np.ndarray,
        field_cartesian: np.ndarray,
    ) -> MACEPolarEFEvaluation:
        positions = torch.tensor(
            np.asarray(positions_angstrom, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        field = torch.tensor(
            np.asarray(field_cartesian, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        energy, source_raw, density = self.conjugate_source_torch(
            positions,
            field,
        )
        (coordinate_gradient,) = torch.autograd.grad(energy, positions)
        arrays = (source_raw, density, coordinate_gradient, field)
        if not all(bool(torch.isfinite(value).all()) for value in arrays):
            raise FloatingPointError("MACE-POLAR-EF evaluation is non-finite.")
        return MACEPolarEFEvaluation(
            energy_ev=float(energy.detach().cpu()),
            conjugate_source_raw=np.asarray(source_raw.detach().cpu(), dtype=float),
            coordinate_gradient_ev_per_angstrom=np.asarray(
                coordinate_gradient.detach().cpu(), dtype=float
            ),
            density_coefficients_diagnostic=np.asarray(
                density.detach().cpu(), dtype=float
            ),
            field_cartesian=np.asarray(field.detach().cpu(), dtype=float),
        )


__all__ = [
    "MACE_POLAR_EF_AUTOGRAD_ORDER",
    "MACE_POLAR_EF_CHECKPOINT_SHA256",
    "MACE_POLAR_EF_MODEL_ID",
    "MACE_POLAR_EF_MAXIMUM_POSITIVE_CURVATURE",
    "MACE_POLAR_EF_PASSIVITY_FIELD_STEP",
    "MACEPolarEFConfig",
    "MACEPolarEFEnergyModel",
    "MACEPolarEFEvaluation",
    "MACEPolarEFResponseAudit",
]

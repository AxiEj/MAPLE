"""Separated source4/native-field8 view of smooth harmonic ddPCM.

The finite-dielectric ddPCM solve uses local reaction-potential coefficients
``X`` rather than an atom-centred field vector.  This module reconstructs the
unique retained harmonic surface-charge coefficients ``q`` from

``K q = X``

with the analytic harmonic single-layer operator ``K``.  The checkpoint-native
two-width Gaussian receiver is then

``u = V_8.T q = V_8.T K^-1 X``.

For point ``l<=1`` sources, the same reconstruction reproduces the ddPCM point
receiver ``C_4 X`` to numerical precision.  This identity is checked at build
time.  Consequently the four-channel solute source and eight-channel model
field remain categorically distinct without inventing an independent receiver
or forcing a square source/field duality.

This is a disabled operational building block.  It exposes no public
E/F/H/V/M capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_DDPCM_V1,
)
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
    SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE,
    BoundaryCoefficientSpace,
    NativeFieldSpace,
    _array_sha256,
    _canonical_sha256,
    _digest,
    _finite_matrix,
    _nonempty,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE, SourceSpace
from maple.solvation.coupling.state_equation import geometry_sha256

from .harmonic_ddpcm_functional import SmoothPartitionHarmonicDDPCMFunctionalCandidate
from .harmonic_ddpcm_primitives import _assemble_point_ddpcm
from .harmonic_torch_primitives import (
    _assemble_gaussian_source,
    _assemble_point_source,
    _assemble_single_layer,
)


HARMONIC_DDPCM_SEPARATED_CONTRACT_ID = (
    "route2-separated-harmonic-ddpcm-point-source4-native-field8-v1"
)
HARMONIC_DDPCM_SEPARATED_CAVITY_PROFILE_ID = (
    "smooth-partition-harmonic-ddpcm-cavity-v1"
)
HARMONIC_DDPCM_SEPARATED_CONFIGURATION_CONTRACT_ID = (
    "smooth-partition-harmonic-ddpcm-point-source-native-receiver-config-v1"
)


@dataclass(frozen=True, slots=True)
class HarmonicDDPCMSeparatedCoordinateKernel:
    """One differentiable Torch graph for ddPCM energy and native field.

    The same graph assembles the finite-dielectric/Schwarz operators, point
    source, harmonic single layer, and two-width Gaussian receiver.  It is the
    production coordinate-derivative implementation; the independent NumPy
    harmonic modules remain numerical references rather than a second force
    path.
    """

    atomic_numbers: tuple[int, ...]
    radii_angstrom: tuple[float, ...]
    transition_width_angstrom2: float
    surface_lmax: int
    partition_lmax: int
    partition_radial_quadrature_order: int
    source_radial_quadrature_order: int
    double_layer_radial_quadrature_order: int
    receiver_radial_quadrature_order: int
    dielectric: float
    runtime_dtype: str
    runtime_device: str
    continuum_configuration_sha256: str

    def __post_init__(self) -> None:
        numbers = tuple(self.atomic_numbers)
        radii = tuple(float(value) for value in self.radii_angstrom)
        if not numbers or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in numbers
        ):
            raise ValueError("atomic_numbers must contain positive integers.")
        if len(radii) != len(numbers) or any(
            not np.isfinite(value) or value <= 0.0 for value in radii
        ):
            raise ValueError("radii_angstrom must be positive and atom-matched.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "radii_angstrom", radii)
        for name in (
            "transition_width_angstrom2",
            "dielectric",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.dielectric <= 1.0:
            raise ValueError("ddPCM coordinate kernel requires dielectric > 1.")
        for name in (
            "surface_lmax",
            "partition_lmax",
            "partition_radial_quadrature_order",
            "source_radial_quadrature_order",
            "double_layer_radial_quadrature_order",
            "receiver_radial_quadrature_order",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.partition_lmax < 2 * self.surface_lmax:
            raise ValueError("partition_lmax must be at least twice surface_lmax.")
        object.__setattr__(
            self,
            "continuum_configuration_sha256",
            _digest(
                self.continuum_configuration_sha256,
                name="continuum_configuration_sha256",
            ),
        )
        object.__setattr__(
            self, "runtime_dtype", _nonempty(self.runtime_dtype, name="runtime_dtype")
        )
        object.__setattr__(
            self,
            "runtime_device",
            _nonempty(self.runtime_device, name="runtime_device"),
        )

    @classmethod
    def from_functional(
        cls,
        functional: SmoothPartitionHarmonicDDPCMFunctionalCandidate,
        *,
        receiver_radial_quadrature_order: int,
    ) -> "HarmonicDDPCMSeparatedCoordinateKernel":
        if not isinstance(functional, SmoothPartitionHarmonicDDPCMFunctionalCandidate):
            raise TypeError(
                "functional must be SmoothPartitionHarmonicDDPCMFunctionalCandidate."
            )
        return cls(
            atomic_numbers=functional.atomic_numbers,
            radii_angstrom=functional.radii_angstrom,
            transition_width_angstrom2=functional.transition_width_angstrom2,
            surface_lmax=functional.surface_lmax,
            partition_lmax=functional.partition_lmax,
            partition_radial_quadrature_order=(
                functional.partition_radial_quadrature_order
            ),
            source_radial_quadrature_order=(
                functional.source_radial_quadrature_order
            ),
            double_layer_radial_quadrature_order=(
                functional.double_layer_radial_quadrature_order
            ),
            receiver_radial_quadrature_order=receiver_radial_quadrature_order,
            dielectric=functional.dielectric,
            runtime_dtype=functional.runtime_dtype,
            runtime_device=functional.runtime_device,
            continuum_configuration_sha256=functional.configuration_sha256(),
        )

    def configuration_sha256(self) -> str:
        directory = Path(__file__).resolve().parent
        return _canonical_sha256(
            {
                "contract": "harmonic-ddpcm-separated-coordinate-kernel-v1",
                "atomic_numbers": list(self.atomic_numbers),
                "radii_angstrom": list(self.radii_angstrom),
                "transition_width_angstrom2": self.transition_width_angstrom2,
                "surface_lmax": self.surface_lmax,
                "partition_lmax": self.partition_lmax,
                "partition_radial_quadrature_order": (
                    self.partition_radial_quadrature_order
                ),
                "source_radial_quadrature_order": (
                    self.source_radial_quadrature_order
                ),
                "double_layer_radial_quadrature_order": (
                    self.double_layer_radial_quadrature_order
                ),
                "receiver_radial_quadrature_order": (
                    self.receiver_radial_quadrature_order
                ),
                "dielectric": self.dielectric,
                "runtime_dtype": self.runtime_dtype,
                "runtime_device": self.runtime_device,
                "continuum_configuration_sha256": (
                    self.continuum_configuration_sha256
                ),
                "implementation_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in (
                        "harmonic_ddpcm_primitives.py",
                        "harmonic_torch_primitives.py",
                        "harmonic_ddpcm_separated.py",
                    )
                },
            }
        )

    def _dtype(self):
        torch = __import__("torch")
        dtype = getattr(torch, self.runtime_dtype.replace("torch.", ""), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("coordinate-kernel runtime dtype is not a Torch dtype.")
        return dtype

    def _positions_tensor(self, positions_angstrom: object, *, requires_grad: bool):
        torch = __import__("torch")
        values = np.asarray(positions_angstrom, dtype=float)
        expected = (len(self.atomic_numbers), 3)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"positions_angstrom must be finite with shape {expected}.")
        return torch.tensor(
            values,
            dtype=self._dtype(),
            device=self.runtime_device,
            requires_grad=requires_grad,
        )

    def _source_tensor(self, source: object, *, reference: object):
        torch = __import__("torch")
        values = ATOMIC_L1_SOURCE_SPACE.validate(
            source, atom_count=len(self.atomic_numbers), name="source"
        )
        return torch.as_tensor(
            values, dtype=reference.dtype, device=reference.device
        )

    def _field_tensor(self, field_cotangent: object, *, reference: object):
        torch = __import__("torch")
        values = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE.validate(
            field_cotangent,
            atom_count=len(self.atomic_numbers),
            name="native field cotangent",
        )
        return torch.as_tensor(
            values, dtype=reference.dtype, device=reference.device
        )

    def _assemble(self, positions: object):
        ddpcm = _assemble_point_ddpcm(
            positions,
            radii=self.radii_angstrom,
            transition_width=self.transition_width_angstrom2,
            lmax=self.surface_lmax,
            partition_lmax=self.partition_lmax,
            partition_radial_order=self.partition_radial_quadrature_order,
            source_radial_order=self.source_radial_quadrature_order,
            double_layer_radial_order=self.double_layer_radial_quadrature_order,
            dielectric=self.dielectric,
        )
        single_layer = _assemble_single_layer(
            positions,
            radii=self.radii_angstrom,
            lmax=self.surface_lmax,
            radial_order=self.receiver_radial_quadrature_order,
        )
        point_surface = _assemble_point_source(
            positions,
            radii=self.radii_angstrom,
            lmax=self.surface_lmax,
            radial_order=self.receiver_radial_quadrature_order,
        )
        gaussian_surface = _assemble_gaussian_source(
            positions,
            radii=self.radii_angstrom,
            lmax=self.surface_lmax,
            radial_order=self.receiver_radial_quadrature_order,
        )
        return ddpcm, single_layer, point_surface, gaussian_surface

    @staticmethod
    def _solve_reaction(ddpcm: object, source_vector: object):
        torch = __import__("torch")
        localized = -(ddpcm.source_operator @ source_vector)
        intermediate = torch.linalg.solve(
            ddpcm.dielectric_operator,
            ddpcm.conductor_operator @ localized,
        )
        reaction = torch.linalg.solve(ddpcm.schwarz_operator, intermediate)
        return localized, intermediate, reaction

    def matrices(self, positions_angstrom: object) -> dict[str, np.ndarray]:
        positions = self._positions_tensor(positions_angstrom, requires_grad=False)
        ddpcm, single_layer, point_surface, gaussian_surface = self._assemble(
            positions
        )
        values = {
            "schwarz_operator": ddpcm.schwarz_operator,
            "conductor_operator": ddpcm.conductor_operator,
            "dielectric_operator": ddpcm.dielectric_operator,
            "source_operator": ddpcm.source_operator,
            "receiver": ddpcm.receiver,
            "single_layer_operator": single_layer,
            "point_surface_source": point_surface,
            "gaussian_surface_source": gaussian_surface,
        }
        return {
            name: np.asarray(value.detach().cpu(), dtype=float).copy()
            for name, value in values.items()
        }

    def continuum_energy_position_gradient(
        self, positions_angstrom: object, source: object
    ) -> np.ndarray:
        torch = __import__("torch")
        positions = self._positions_tensor(positions_angstrom, requires_grad=True)
        source_tensor = self._source_tensor(source, reference=positions)
        ddpcm, _single_layer, _point_surface, _gaussian_surface = self._assemble(
            positions
        )
        _localized, _intermediate, reaction = self._solve_reaction(
            ddpcm, source_tensor.reshape(-1)
        )
        scalar = 0.5 * (
            source_tensor.reshape(-1) @ (ddpcm.receiver @ reaction)
        )
        (gradient,) = torch.autograd.grad(scalar, (positions,), create_graph=False)
        result = np.asarray(gradient.detach().cpu(), dtype=float).copy()
        if result.shape != (len(self.atomic_numbers), 3) or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("ddPCM energy position gradient is invalid.")
        return result

    def source_field_position_vjp(
        self,
        positions_angstrom: object,
        source: object,
        field_cotangent: object,
    ) -> np.ndarray:
        torch = __import__("torch")
        positions = self._positions_tensor(positions_angstrom, requires_grad=True)
        source_tensor = self._source_tensor(source, reference=positions)
        field_tensor = self._field_tensor(field_cotangent, reference=positions)
        ddpcm, single_layer, _point_surface, gaussian_surface = self._assemble(
            positions
        )
        _localized, _intermediate, reaction = self._solve_reaction(
            ddpcm, source_tensor.reshape(-1)
        )
        surface_charge = torch.linalg.solve(single_layer, reaction)
        native_field = gaussian_surface.T @ surface_charge
        scalar = field_tensor.reshape(-1) @ native_field
        (gradient,) = torch.autograd.grad(scalar, (positions,), create_graph=False)
        result = np.asarray(gradient.detach().cpu(), dtype=float).copy()
        if result.shape != (len(self.atomic_numbers), 3) or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("ddPCM native-field position VJP is invalid.")
        return result


def _readonly_matrix(values: object, *, name: str) -> np.ndarray:
    result = _finite_matrix(values, name=name)
    result.setflags(write=False)
    return result


class HarmonicDDPCMSeparatedSnapshot:
    """Immutable geometry-bound composite ddPCM response snapshot."""

    __slots__ = (
        "_boundary_dimension",
        "_configuration_sha256",
        "_continuum_provenance_sha256",
        "_coordinate_kernel",
        "_energy_hessian_values",
        "_energy_receiver_values",
        "_field_map_values",
        "_gaussian_surface_source_values",
        "_point_surface_source_values",
        "_positions_angstrom",
        "_provenance_sha256",
        "_reaction_map_values",
        "_reaction_to_native_field_values",
        "_sealed",
        "_single_layer_values",
        "_source_embedding_values",
        "_source_to_local_values",
        "_surface_conductor_values",
        "_surface_dielectric_values",
        "_surface_schwarz_values",
        "atom_count",
        "boundary_space",
        "cavity_profile_id",
        "configuration_contract_id",
        "continuum_configuration_sha256",
        "continuum_profile_id",
        "continuum_provider_id",
        "coupling_id",
        "geometry_sha256",
        "point_receiver_reconstruction_max_abs_error",
        "receiver_space",
        "scalar_id",
        "source_space",
        "topology_sha256",
    )

    contract_id = HARMONIC_DDPCM_SEPARATED_CONTRACT_ID
    capabilities = ()
    linear_response = True
    model_field_is_energy_gradient = False
    structurally_rotation_equivariant = True

    def __init__(
        self,
        *,
        atom_count: int,
        geometry_digest: str,
        continuum_provider_id: str,
        continuum_profile_id: str,
        continuum_configuration_sha256: str,
        continuum_provenance_sha256: str,
        topology_sha256: str,
        positions_angstrom: object,
        coordinate_kernel: HarmonicDDPCMSeparatedCoordinateKernel,
        schwarz_operator: object,
        conductor_operator: object,
        dielectric_operator: object,
        source_to_local_potential: object,
        energy_receiver: object,
        single_layer_operator: object,
        point_surface_source: object,
        gaussian_surface_source: object,
        reaction_to_native_field: object,
        source_embedding: object,
        scalar_id: str = (
            OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_DDPCM_V1
        ),
        source_space: SourceSpace = ATOMIC_L1_SOURCE_SPACE,
        receiver_space: NativeFieldSpace = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
        boundary_space: BoundaryCoefficientSpace = (
            SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE
        ),
        coupling_id: str = SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
    ) -> None:
        if type(atom_count) is not int or atom_count < 1:
            raise ValueError("atom_count must be a positive integer.")
        if not isinstance(source_space, SourceSpace):
            raise TypeError("source_space must be SourceSpace.")
        if not isinstance(receiver_space, NativeFieldSpace):
            raise TypeError("receiver_space must be NativeFieldSpace.")
        if not isinstance(boundary_space, BoundaryCoefficientSpace):
            raise TypeError("boundary_space must be BoundaryCoefficientSpace.")
        if not isinstance(
            coordinate_kernel, HarmonicDDPCMSeparatedCoordinateKernel
        ):
            raise TypeError(
                "coordinate_kernel must be HarmonicDDPCMSeparatedCoordinateKernel."
            )
        positions = np.asarray(positions_angstrom, dtype=float)
        if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                f"positions_angstrom must be finite with shape ({atom_count}, 3)."
            )
        if len(coordinate_kernel.atomic_numbers) != atom_count:
            raise ValueError("coordinate kernel atom count does not match snapshot.")
        if coordinate_kernel.continuum_configuration_sha256 != _digest(
            continuum_configuration_sha256,
            name="continuum_configuration_sha256",
        ):
            raise ValueError(
                "coordinate kernel and continuum configuration do not match."
            )

        schwarz = _readonly_matrix(schwarz_operator, name="Schwarz operator")
        conductor = _readonly_matrix(conductor_operator, name="conductor operator")
        dielectric = _readonly_matrix(
            dielectric_operator, name="finite-dielectric operator"
        )
        source = _readonly_matrix(
            source_to_local_potential, name="point source-to-local-potential operator"
        )
        energy_receiver_matrix = _readonly_matrix(
            energy_receiver, name="point energy receiver"
        )
        single_layer = _readonly_matrix(
            single_layer_operator, name="harmonic single-layer operator"
        )
        point_surface = _readonly_matrix(
            point_surface_source, name="point full-sphere source operator"
        )
        gaussian_surface = _readonly_matrix(
            gaussian_surface_source, name="Gaussian full-sphere source operator"
        )
        native_receiver = _readonly_matrix(
            reaction_to_native_field, name="reaction-to-native-field operator"
        )
        embedding = _readonly_matrix(source_embedding, name="source embedding")

        boundary_dimension = schwarz.shape[0]
        source_dimension = atom_count * source_space.component_count
        receiver_dimension = atom_count * receiver_space.component_count
        square_shape = (boundary_dimension, boundary_dimension)
        if any(
            matrix.shape != square_shape
            for matrix in (schwarz, conductor, dielectric, single_layer)
        ):
            raise ValueError("all ddPCM boundary operators must share one square shape.")
        if source.shape != (boundary_dimension, source_dimension):
            raise ValueError("source-to-local operator has an incompatible shape.")
        if energy_receiver_matrix.shape != (source_dimension, boundary_dimension):
            raise ValueError("energy receiver has an incompatible shape.")
        if point_surface.shape != (boundary_dimension, source_dimension):
            raise ValueError("point full-sphere source has an incompatible shape.")
        if gaussian_surface.shape != (boundary_dimension, receiver_dimension):
            raise ValueError("Gaussian full-sphere source has an incompatible shape.")
        if native_receiver.shape != (receiver_dimension, boundary_dimension):
            raise ValueError("native receiver has an incompatible shape.")
        if embedding.shape != (receiver_dimension, source_dimension):
            raise ValueError("source embedding has an incompatible shape.")
        if np.linalg.matrix_rank(embedding) != source_dimension:
            raise ValueError("source embedding must have full source-column rank.")

        for name, matrix in (
            ("Schwarz", schwarz),
            ("finite-dielectric", dielectric),
            ("single-layer", single_layer),
        ):
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            if singular_values[-1] <= 1.0e-12 or not np.isfinite(
                singular_values[0] / singular_values[-1]
            ):
                raise ValueError(f"{name} operator failed its invertibility gate.")
        if not np.allclose(
            single_layer, single_layer.T, rtol=0.0, atol=2.0e-12
        ):
            raise ValueError("harmonic single-layer operator must be symmetric.")

        reaction_map = np.linalg.solve(
            schwarz,
            np.linalg.solve(dielectric, conductor @ (-source)),
        )
        point_reconstructed_receiver = np.linalg.solve(
            single_layer.T, point_surface
        ).T
        # The local-potential coefficients contain buried/Schwarz directions
        # that are not physical surface-charge states.  Receiver equivalence
        # is therefore required on the reachable ddPCM reaction subspace, not
        # on arbitrary coefficient vectors outside the solve image.
        reconstruction_error = float(
            np.max(
                np.abs(
                    (point_reconstructed_receiver - energy_receiver_matrix)
                    @ reaction_map
                )
            )
        )
        if not np.isfinite(reconstruction_error) or reconstruction_error > 3.0e-11:
            raise ValueError(
                "harmonic surface-charge reconstruction does not reproduce the "
                "point receiver on the reachable ddPCM reaction subspace."
            )
        field_map = native_receiver @ reaction_map
        energy_hessian = energy_receiver_matrix @ reaction_map
        scale = max(1.0, float(np.linalg.norm(energy_hessian, ord="fro")))
        if not np.allclose(
            energy_hessian,
            energy_hessian.T,
            rtol=0.0,
            atol=3.0e-11 * scale,
        ):
            raise ValueError("ddPCM source-energy Hessian lost reciprocity.")
        object.__setattr__(self, "atom_count", atom_count)
        object.__setattr__(
            self, "geometry_sha256", _digest(geometry_digest, name="geometry_digest")
        )
        object.__setattr__(
            self,
            "continuum_provider_id",
            _nonempty(continuum_provider_id, name="continuum_provider_id"),
        )
        object.__setattr__(
            self,
            "continuum_profile_id",
            _nonempty(continuum_profile_id, name="continuum_profile_id"),
        )
        object.__setattr__(
            self,
            "cavity_profile_id",
            HARMONIC_DDPCM_SEPARATED_CAVITY_PROFILE_ID,
        )
        object.__setattr__(
            self,
            "configuration_contract_id",
            HARMONIC_DDPCM_SEPARATED_CONFIGURATION_CONTRACT_ID,
        )
        object.__setattr__(self, "scalar_id", _nonempty(scalar_id, name="scalar_id"))
        object.__setattr__(
            self,
            "continuum_configuration_sha256",
            _digest(
                continuum_configuration_sha256,
                name="continuum_configuration_sha256",
            ),
        )
        object.__setattr__(
            self,
            "_continuum_provenance_sha256",
            _digest(
                continuum_provenance_sha256,
                name="continuum_provenance_sha256",
            ),
        )
        object.__setattr__(
            self, "topology_sha256", _digest(topology_sha256, name="topology_sha256")
        )
        object.__setattr__(self, "coupling_id", _nonempty(coupling_id, name="coupling_id"))
        object.__setattr__(self, "source_space", source_space)
        object.__setattr__(self, "receiver_space", receiver_space)
        object.__setattr__(self, "boundary_space", boundary_space)
        object.__setattr__(
            self,
            "_positions_angstrom",
            tuple(tuple(float(value) for value in row) for row in positions),
        )
        object.__setattr__(self, "_coordinate_kernel", coordinate_kernel)
        object.__setattr__(self, "_boundary_dimension", boundary_dimension)
        object.__setattr__(
            self,
            "point_receiver_reconstruction_max_abs_error",
            reconstruction_error,
        )
        for name, matrix in (
            ("_surface_schwarz_values", schwarz),
            ("_surface_conductor_values", conductor),
            ("_surface_dielectric_values", dielectric),
            ("_source_to_local_values", source),
            ("_energy_receiver_values", energy_receiver_matrix),
            ("_single_layer_values", single_layer),
            ("_point_surface_source_values", point_surface),
            ("_gaussian_surface_source_values", gaussian_surface),
            ("_reaction_to_native_field_values", native_receiver),
            ("_source_embedding_values", embedding),
            ("_reaction_map_values", reaction_map),
            ("_field_map_values", field_map),
            ("_energy_hessian_values", energy_hessian),
        ):
            object.__setattr__(self, name, tuple(float(value) for value in matrix.flat))
        configuration = self._current_configuration_sha256()
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "_provenance_sha256",
            _canonical_sha256(
                {
                    "contract_id": self.contract_id,
                    "configuration_sha256": configuration,
                    "continuum_provenance_sha256": (
                        self._continuum_provenance_sha256
                    ),
                    "implementation_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "capabilities": "none",
                }
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("HarmonicDDPCMSeparatedSnapshot is immutable.")
        object.__setattr__(self, name, value)

    @property
    def boundary_dimension(self) -> int:
        return self._boundary_dimension

    @property
    def source_dimension(self) -> int:
        return self.atom_count * self.source_space.component_count

    @property
    def receiver_dimension(self) -> int:
        return self.atom_count * self.receiver_space.component_count

    @staticmethod
    def _matrix(values: tuple[float, ...], rows: int, columns: int) -> np.ndarray:
        result = np.asarray(values, dtype=float).reshape(rows, columns).copy()
        result.setflags(write=False)
        return result

    @property
    def schwarz_operator(self) -> np.ndarray:
        return self._matrix(
            self._surface_schwarz_values,
            self.boundary_dimension,
            self.boundary_dimension,
        )

    @property
    def conductor_operator(self) -> np.ndarray:
        return self._matrix(
            self._surface_conductor_values,
            self.boundary_dimension,
            self.boundary_dimension,
        )

    @property
    def dielectric_operator(self) -> np.ndarray:
        return self._matrix(
            self._surface_dielectric_values,
            self.boundary_dimension,
            self.boundary_dimension,
        )

    @property
    def source_to_boundary(self) -> np.ndarray:
        return self._matrix(
            self._source_to_local_values,
            self.boundary_dimension,
            self.source_dimension,
        )

    @property
    def energy_receiver(self) -> np.ndarray:
        return self._matrix(
            self._energy_receiver_values,
            self.source_dimension,
            self.boundary_dimension,
        )

    @property
    def single_layer_operator(self) -> np.ndarray:
        return self._matrix(
            self._single_layer_values,
            self.boundary_dimension,
            self.boundary_dimension,
        )

    @property
    def point_surface_source(self) -> np.ndarray:
        return self._matrix(
            self._point_surface_source_values,
            self.boundary_dimension,
            self.source_dimension,
        )

    @property
    def gaussian_surface_source(self) -> np.ndarray:
        return self._matrix(
            self._gaussian_surface_source_values,
            self.boundary_dimension,
            self.receiver_dimension,
        )

    @property
    def boundary_to_native_field(self) -> np.ndarray:
        return self._matrix(
            self._reaction_to_native_field_values,
            self.receiver_dimension,
            self.boundary_dimension,
        )

    @property
    def source_embedding(self) -> np.ndarray:
        return self._matrix(
            self._source_embedding_values,
            self.receiver_dimension,
            self.source_dimension,
        )

    @property
    def reaction_map(self) -> np.ndarray:
        return self._matrix(
            self._reaction_map_values,
            self.boundary_dimension,
            self.source_dimension,
        )

    @property
    def source_to_native_field(self) -> np.ndarray:
        return self._matrix(
            self._field_map_values,
            self.receiver_dimension,
            self.source_dimension,
        )

    @property
    def energy_hessian(self) -> np.ndarray:
        return self._matrix(
            self._energy_hessian_values,
            self.source_dimension,
            self.source_dimension,
        )

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._provenance_sha256

    @property
    def positions_angstrom(self) -> np.ndarray:
        result = np.asarray(self._positions_angstrom, dtype=float).copy()
        result.setflags(write=False)
        return result

    @property
    def coordinate_kernel(self) -> HarmonicDDPCMSeparatedCoordinateKernel:
        return self._coordinate_kernel

    def _current_configuration_sha256(self) -> str:
        return _canonical_sha256(
            {
                "contract_id": self.contract_id,
                "coupling_id": self.coupling_id,
                "atom_count": self.atom_count,
                "geometry_sha256": self.geometry_sha256,
                "continuum_provider_id": self.continuum_provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "configuration_contract_id": self.configuration_contract_id,
                "scalar_id": self.scalar_id,
                "continuum_configuration_sha256": (
                    self.continuum_configuration_sha256
                ),
                "continuum_provenance_sha256": (
                    self._continuum_provenance_sha256
                ),
                "topology_sha256": self.topology_sha256,
                "positions_angstrom_sha256": _array_sha256(
                    self.positions_angstrom, name="positions_angstrom"
                ),
                "coordinate_kernel_sha256": (
                    self.coordinate_kernel.configuration_sha256()
                ),
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "boundary_space_sha256": self.boundary_space.metadata_hash(),
                "schwarz_operator_sha256": _array_sha256(
                    self.schwarz_operator, name="Schwarz operator"
                ),
                "conductor_operator_sha256": _array_sha256(
                    self.conductor_operator, name="conductor operator"
                ),
                "dielectric_operator_sha256": _array_sha256(
                    self.dielectric_operator, name="dielectric operator"
                ),
                "source_to_local_sha256": _array_sha256(
                    self.source_to_boundary, name="source-to-local operator"
                ),
                "energy_receiver_sha256": _array_sha256(
                    self.energy_receiver, name="energy receiver"
                ),
                "single_layer_sha256": _array_sha256(
                    self.single_layer_operator, name="single-layer operator"
                ),
                "point_surface_source_sha256": _array_sha256(
                    self.point_surface_source, name="point surface source"
                ),
                "gaussian_surface_source_sha256": _array_sha256(
                    self.gaussian_surface_source, name="Gaussian surface source"
                ),
                "reaction_to_native_field_sha256": _array_sha256(
                    self.boundary_to_native_field, name="native receiver"
                ),
                "source_embedding_sha256": _array_sha256(
                    self.source_embedding, name="source embedding"
                ),
                "point_receiver_reconstruction_max_abs_error": (
                    self.point_receiver_reconstruction_max_abs_error
                ),
                "state_equations": "A_eps G=A_inf(-B c); L X=G",
                "native_field_equation": "K q=X; u=V8.T q",
                "continuum_energy": "G=1/2 c.T C4 X",
                "source_receiver_duality_assumed": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("harmonic ddPCM separated configuration drifted.")
        return current

    def source_rhs(self, source: object) -> np.ndarray:
        values = self.source_space.validate(source, atom_count=self.atom_count)
        self.configuration_sha256()
        return self.source_to_boundary @ values.reshape(-1)

    def solve_boundary(self, source: object) -> np.ndarray:
        values = self.source_space.validate(source, atom_count=self.atom_count)
        return self.reaction_map @ values.reshape(-1)

    def native_field_from_boundary(self, boundary_state: object) -> np.ndarray:
        reaction = self.boundary_space.validate(
            boundary_state, dimension=self.boundary_dimension
        )
        field = self.boundary_to_native_field @ reaction
        return self.receiver_space.validate(
            field.reshape(self.receiver_space.shape(self.atom_count)),
            atom_count=self.atom_count,
            name="checkpoint-native harmonic ddPCM field",
        )

    def native_field(self, source: object) -> np.ndarray:
        values = self.source_space.validate(source, atom_count=self.atom_count)
        field = self.source_to_native_field @ values.reshape(-1)
        return self.receiver_space.validate(
            field.reshape(self.receiver_space.shape(self.atom_count)),
            atom_count=self.atom_count,
            name="checkpoint-native harmonic ddPCM field",
        )

    def source_field_jvp(self, source_direction: object) -> np.ndarray:
        return self.native_field(source_direction)

    def source_field_vjp(self, field_cotangent: object) -> np.ndarray:
        cotangent = self.receiver_space.validate(
            field_cotangent,
            atom_count=self.atom_count,
            name="native field cotangent",
        )
        result = (self.source_to_native_field.T @ cotangent.reshape(-1)).reshape(
            self.source_space.shape(self.atom_count)
        )
        return self.source_space.validate(
            result,
            atom_count=self.atom_count,
            name="harmonic ddPCM source-field VJP",
        )

    def continuum_energy_eV(self, source: object) -> float:
        values = self.source_space.validate(source, atom_count=self.atom_count)
        vector = values.reshape(-1)
        result = 0.5 * float(vector @ (self.energy_hessian @ vector))
        if not np.isfinite(result):
            raise RuntimeError("harmonic ddPCM continuum energy is non-finite.")
        return result

    def continuum_source_gradient(self, source: object) -> np.ndarray:
        values = self.source_space.validate(source, atom_count=self.atom_count)
        symmetric = 0.5 * (self.energy_hessian + self.energy_hessian.T)
        result = (symmetric @ values.reshape(-1)).reshape(
            self.source_space.shape(self.atom_count)
        )
        return self.source_space.validate(
            result,
            atom_count=self.atom_count,
            name="harmonic ddPCM continuum source gradient",
        )

    def continuum_energy_position_gradient(self, source: object) -> np.ndarray:
        self.configuration_sha256()
        return self.coordinate_kernel.continuum_energy_position_gradient(
            self.positions_angstrom,
            self.source_space.validate(source, atom_count=self.atom_count),
        )

    def source_field_position_vjp(
        self, source: object, field_cotangent: object
    ) -> np.ndarray:
        self.configuration_sha256()
        return self.coordinate_kernel.source_field_position_vjp(
            self.positions_angstrom,
            self.source_space.validate(source, atom_count=self.atom_count),
            self.receiver_space.validate(
                field_cotangent,
                atom_count=self.atom_count,
                name="native field cotangent",
            ),
        )


def build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
    continuum: SmoothPartitionHarmonicDDPCMFunctionalCandidate,
    geometry: object,
    *,
    receiver_radial_quadrature_order: int = 128,
    scalar_id: str = (
        OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_DDPCM_V1
    ),
    coupling_id: str = SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
) -> HarmonicDDPCMSeparatedSnapshot:
    """Build a point-source4/ddPCM/native-Gaussian-field8 snapshot."""

    if not isinstance(
        continuum, SmoothPartitionHarmonicDDPCMFunctionalCandidate
    ):
        raise TypeError(
            "continuum must be SmoothPartitionHarmonicDDPCMFunctionalCandidate."
        )
    positions = np.asarray(getattr(geometry, "positions", geometry), dtype=float)
    atom_count = len(continuum.radii_angstrom)
    if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"geometry positions must be finite with shape ({atom_count}, 3)."
        )
    if type(receiver_radial_quadrature_order) is not int or not (
        1 <= receiver_radial_quadrature_order <= 4096
    ):
        raise ValueError(
            "receiver_radial_quadrature_order must be an integer in [1, 4096]."
        )

    matrices = continuum.debug_geometry_matrices(geometry)
    lmax = continuum.surface_lmax
    coordinate_kernel = HarmonicDDPCMSeparatedCoordinateKernel.from_functional(
        continuum,
        receiver_radial_quadrature_order=receiver_radial_quadrature_order,
    )
    coordinate_matrices = coordinate_kernel.matrices(positions)
    for name in (
        "schwarz_operator",
        "conductor_operator",
        "dielectric_operator",
        "source_operator",
        "receiver",
    ):
        if not np.allclose(
            coordinate_matrices[name], matrices[name], rtol=0.0, atol=3.0e-12
        ):
            raise RuntimeError(
                f"coordinate-kernel {name} does not reproduce the scalar graph."
            )
    single_layer = coordinate_matrices["single_layer_operator"]
    point_surface = coordinate_matrices["point_surface_source"]
    gaussian_surface = coordinate_matrices["gaussian_surface_source"]
    gaussian_receiver = np.linalg.solve(single_layer.T, gaussian_surface).T
    energy_receiver = np.asarray(matrices["receiver"], dtype=float)
    embedding = np.kron(
        np.eye(atom_count), mace_polar_learned_source_embedding_matrix()
    )
    return HarmonicDDPCMSeparatedSnapshot(
        atom_count=atom_count,
        geometry_digest=geometry_sha256(geometry),
        continuum_provider_id=continuum.provider_id,
        continuum_profile_id=continuum.continuum_profile_id,
        continuum_configuration_sha256=continuum.configuration_sha256(),
        continuum_provenance_sha256=continuum.provenance_sha256,
        topology_sha256=_canonical_sha256(
            {
                "contract_id": HARMONIC_DDPCM_SEPARATED_CONTRACT_ID,
                "continuum_configuration_sha256": continuum.configuration_sha256(),
                "atom_count": atom_count,
                "surface_lmax": lmax,
                "coefficient_order": (
                    "atom-major/l-ascending/m-ascending/real-harmonic"
                ),
                "fixed_coefficient_dimension": int(
                    np.asarray(matrices["schwarz_operator"]).shape[0]
                ),
                "active_set": "none-fixed-coefficient-topology",
            }
        ),
        positions_angstrom=positions,
        coordinate_kernel=coordinate_kernel,
        schwarz_operator=matrices["schwarz_operator"],
        conductor_operator=matrices["conductor_operator"],
        dielectric_operator=matrices["dielectric_operator"],
        source_to_local_potential=matrices["source_operator"],
        energy_receiver=energy_receiver,
        single_layer_operator=single_layer,
        point_surface_source=point_surface,
        gaussian_surface_source=gaussian_surface,
        reaction_to_native_field=gaussian_receiver,
        source_embedding=embedding,
        scalar_id=scalar_id,
        coupling_id=coupling_id,
    )


__all__ = [
    "HarmonicDDPCMSeparatedCoordinateKernel",
    "HARMONIC_DDPCM_SEPARATED_CAVITY_PROFILE_ID",
    "HARMONIC_DDPCM_SEPARATED_CONFIGURATION_CONTRACT_ID",
    "HARMONIC_DDPCM_SEPARATED_CONTRACT_ID",
    "HarmonicDDPCMSeparatedSnapshot",
    "build_mace_polar_point_harmonic_ddpcm_separated_snapshot",
]

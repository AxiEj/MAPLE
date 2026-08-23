"""General-source harmonic ddPCM for MDP-permanent/POLAR-induced coupling.

The permanent and induced branches are a physical direct sum, not a shared
source-kernel alias.  For permanent point multipoles ``p`` and the induced
MACE-POLAR first-width Gaussian multipoles ``d`` the two ddPCM source objects
are

``psi = p + d``
``phi = B_point p + B_GTO(1.5 A) d``.

Writing

``N = L^-1 A_epsilon^-1 A_infinity``

for the finite-dielectric harmonic solve, the primal and adjoint coefficient
states are

``X  = -N phi``
``xi = N.T C_point.T psi``.

The operational electrostatic scalar and the checkpoint-native external-MEP
drive are deliberately distinct:

``G = 0.5 psi.T C_point X``
``u = -B_native.T xi``.

This is the coefficient-space analogue of ddX's general-source ``psi/phi``
formulation.  It is essential for normalized Gaussian sources, whose tails
extend outside every finite cavity: the energy derivative contains both the
``psi`` and ``phi`` contributions, while the conventional model drive is the
complete ``phi``-side adjoint contraction.  No full-sphere surface-charge
reconstruction is assumed, so overlapping exposed sphere charts remain valid.

Every moving-geometry derivative is generated from the same Torch assembly
and the same primal/adjoint contractions.  This module contains no
model-specific fixed-point logic and admits no public E/F/H/V/M capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2,
)
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    _canonical_sha256,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.coupling.state_equation import geometry_sha256

from .harmonic_ddpcm_functional import SmoothPartitionHarmonicDDPCMFunctionalCandidate
from .harmonic_ddpcm_primitives import _assemble_point_ddpcm
from .harmonic_ddpcm_separated import HarmonicDDPCMSeparatedCoordinateKernel
from .harmonic_torch_primitives import (
    _assemble_gaussian_source,
    _weighted_basis_block,
)


HARMONIC_DDPCM_HYBRID_COUPLING_ID = (
    "route2-coupling-macemdppoint-macepolarinduced-gto1p5-"
    "smoothharmonic-ddpcm-nativefield8-v2"
)
HARMONIC_DDPCM_HYBRID_CONTRACT_ID = (
    "route2-harmonic-ddpcm-general-source-psi-phi-primal-adjoint-v2"
)
HARMONIC_DDPCM_HYBRID_PROFILE_ID = (
    "smooth-partition-harmonic-ddpcm-hybrid-general-source-v2"
)
HARMONIC_DDPCM_HYBRID_CAVITY_PROFILE_ID = (
    "smooth-partition-harmonic-ddpcm-cavity-v1"
)


def _readonly_matrix(
    values: object, *, shape: tuple[int, int], name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size != shape[0] * shape[1] or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(array, dtype=np.float64, copy=True).reshape(shape)
    result.setflags(write=False)
    return result


def _matrix_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _localize_gaussian_native_source(
    positions: object,
    *,
    assembly: object,
    radii: tuple[float, ...],
    surface_lmax: int,
    partition_lmax: int,
    radial_order: int,
):
    """Project the full eight-channel Gaussian MEP through the partition."""

    atom_count = len(radii)
    surface_dimension = (surface_lmax + 1) ** 2
    physical_lmax = surface_lmax + partition_lmax
    physical_dimension = (physical_lmax + 1) ** 2
    raw = _assemble_gaussian_source(
        positions,
        radii=radii,
        lmax=physical_lmax,
        radial_order=radial_order,
    )
    localized = positions.new_zeros(
        (atom_count * surface_dimension, atom_count * 8)
    )
    for target, (exposed, _overlap) in enumerate(assembly.partition):
        weighted_test = _weighted_basis_block(
            exposed,
            exposure_lmax=partition_lmax,
            basis_lmax=surface_lmax,
        )
        raw_block = raw[
            target * physical_dimension : (target + 1) * physical_dimension
        ]
        localized[
            target * surface_dimension : (target + 1) * surface_dimension
        ] = weighted_test.T @ raw_block
    return localized


@dataclass(frozen=True, slots=True)
class HarmonicDDPCMHybridCoordinateKernel:
    """Same-graph primal/adjoint coordinate contractions."""

    base: HarmonicDDPCMSeparatedCoordinateKernel

    def __post_init__(self) -> None:
        if not isinstance(self.base, HarmonicDDPCMSeparatedCoordinateKernel):
            raise TypeError("base must be HarmonicDDPCMSeparatedCoordinateKernel.")
        self.base.configuration_sha256()

    @classmethod
    def from_functional(
        cls,
        functional: SmoothPartitionHarmonicDDPCMFunctionalCandidate,
        *,
        receiver_radial_quadrature_order: int,
    ) -> "HarmonicDDPCMHybridCoordinateKernel":
        return cls(
            HarmonicDDPCMSeparatedCoordinateKernel.from_functional(
                functional,
                receiver_radial_quadrature_order=receiver_radial_quadrature_order,
            )
        )

    @property
    def atom_count(self) -> int:
        return len(self.base.atomic_numbers)

    def configuration_sha256(self) -> str:
        return _canonical_sha256(
            {
                "contract": HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
                "base_coordinate_kernel_sha256": self.base.configuration_sha256(),
                "permanent_phi": "exterior-point-l1",
                "induced_phi": "normalized-Gaussian-l1-sigma1p5-angstrom",
                "shared_psi": "point-multipole-moments-p-plus-d",
                "native_receiver": (
                    "phi-adjoint-normalized-Gaussian-l1-sigma1p5-and-3p0-angstrom"
                ),
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )

    def _source_tensor(self, source: object, *, reference: object):
        values = ATOMIC_L1_SOURCE_SPACE.validate(
            source, atom_count=self.atom_count, name="atomic l<=1 source"
        )
        return __import__("torch").as_tensor(
            values, dtype=reference.dtype, device=reference.device
        )

    def _field_tensor(self, field: object, *, reference: object):
        values = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE.validate(
            field, atom_count=self.atom_count, name="native field cotangent"
        )
        return __import__("torch").as_tensor(
            values, dtype=reference.dtype, device=reference.device
        )

    def _assemble(self, positions: object):
        ddpcm = _assemble_point_ddpcm(
            positions,
            radii=self.base.radii_angstrom,
            transition_width=self.base.transition_width_angstrom2,
            lmax=self.base.surface_lmax,
            partition_lmax=self.base.partition_lmax,
            partition_radial_order=self.base.partition_radial_quadrature_order,
            source_radial_order=self.base.source_radial_quadrature_order,
            double_layer_radial_order=(
                self.base.double_layer_radial_quadrature_order
            ),
            dielectric=self.base.dielectric,
        )
        gaussian_native = _localize_gaussian_native_source(
            positions,
            assembly=ddpcm,
            radii=self.base.radii_angstrom,
            surface_lmax=self.base.surface_lmax,
            partition_lmax=self.base.partition_lmax,
            radial_order=self.base.source_radial_quadrature_order,
        )
        embedding = positions.new_tensor(
            np.kron(
                np.eye(self.atom_count),
                mace_polar_learned_source_embedding_matrix(),
            )
        )
        gaussian_induced = gaussian_native @ embedding
        return ddpcm, gaussian_native, gaussian_induced, embedding

    @staticmethod
    def _solve_general(
        ddpcm: object,
        gaussian_native: object,
        gaussian_induced: object,
        permanent: object,
        induced: object,
    ):
        torch = __import__("torch")
        psi = permanent + induced
        phi = ddpcm.source_operator @ permanent + gaussian_induced @ induced
        localized = -phi
        intermediate = torch.linalg.solve(
            ddpcm.dielectric_operator,
            ddpcm.conductor_operator @ localized,
        )
        reaction = torch.linalg.solve(ddpcm.schwarz_operator, intermediate)

        # xi = N.T C.T psi, evaluated as three transpose contractions rather
        # than by forming N or differentiating a matrix inverse.
        schwarz_adjoint = torch.linalg.solve(
            ddpcm.schwarz_operator.T,
            ddpcm.receiver.T @ psi,
        )
        dielectric_adjoint = torch.linalg.solve(
            ddpcm.dielectric_operator.T,
            schwarz_adjoint,
        )
        adjoint = ddpcm.conductor_operator.T @ dielectric_adjoint
        field = -(gaussian_native.T @ adjoint)
        energy = 0.5 * (psi @ (ddpcm.receiver @ reaction))
        return energy, field, reaction, adjoint

    def matrices(self, positions_angstrom: object) -> dict[str, np.ndarray]:
        positions = self.base._positions_tensor(
            positions_angstrom, requires_grad=False
        )
        ddpcm, gaussian_native, gaussian_induced, embedding = self._assemble(
            positions
        )
        values = {
            "schwarz_operator": ddpcm.schwarz_operator,
            "conductor_operator": ddpcm.conductor_operator,
            "dielectric_operator": ddpcm.dielectric_operator,
            "point_local_source": ddpcm.source_operator,
            "point_energy_receiver": ddpcm.receiver,
            "gaussian_native_local_source": gaussian_native,
            "gaussian_induced_local_source": gaussian_induced,
            "source_embedding": embedding,
        }
        return {
            name: np.asarray(value.detach().cpu(), dtype=float).copy()
            for name, value in values.items()
        }

    def _scalar_and_field(
        self,
        positions: object,
        permanent_source: object,
        induced_source: object,
    ):
        permanent = self._source_tensor(
            permanent_source, reference=positions
        ).reshape(-1)
        induced = self._source_tensor(induced_source, reference=positions).reshape(
            -1
        )
        ddpcm, gaussian_native, gaussian_induced, _embedding = self._assemble(
            positions
        )
        return self._solve_general(
            ddpcm, gaussian_native, gaussian_induced, permanent, induced
        )[:2]

    def continuum_energy_position_gradient(
        self,
        positions_angstrom: object,
        permanent_source: object,
        induced_source: object,
    ) -> np.ndarray:
        torch = __import__("torch")
        positions = self.base._positions_tensor(
            positions_angstrom, requires_grad=True
        )
        energy, _field = self._scalar_and_field(
            positions, permanent_source, induced_source
        )
        (gradient,) = torch.autograd.grad(energy, (positions,), create_graph=False)
        result = np.asarray(gradient.detach().cpu(), dtype=float).copy()
        if result.shape != (self.atom_count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError("hybrid ddPCM energy coordinate gradient is invalid.")
        return result

    def model_field_position_vjp(
        self,
        positions_angstrom: object,
        permanent_source: object,
        induced_source: object,
        field_cotangent: object,
    ) -> np.ndarray:
        torch = __import__("torch")
        positions = self.base._positions_tensor(
            positions_angstrom, requires_grad=True
        )
        _energy, field = self._scalar_and_field(
            positions, permanent_source, induced_source
        )
        cotangent = self._field_tensor(
            field_cotangent, reference=positions
        ).reshape(-1)
        scalar = cotangent @ field
        (gradient,) = torch.autograd.grad(scalar, (positions,), create_graph=False)
        result = np.asarray(gradient.detach().cpu(), dtype=float).copy()
        if result.shape != (self.atom_count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError("hybrid ddPCM field coordinate VJP is invalid.")
        return result


class HarmonicDDPCMHybridSnapshot:
    """Immutable geometry-bound general-source ddPCM response."""

    __slots__ = (
        "_adjoint_total_map_values",
        "_combined_hessian_values",
        "_configuration_sha256",
        "_coordinate_kernel",
        "_energy_receiver_values",
        "_forward_boundary_dimension",
        "_gaussian_induced_local_values",
        "_gaussian_native_local_values",
        "_induced_reaction_map_values",
        "_native_field_total_map_values",
        "_permanent_reaction_map_values",
        "_point_local_values",
        "_positions_angstrom",
        "_provenance_sha256",
        "_raw_hessian_asymmetry",
        "_sealed",
        "_surface_conductor_values",
        "_surface_dielectric_values",
        "_surface_schwarz_values",
        "atom_count",
        "cavity_profile_id",
        "continuum_configuration_sha256",
        "continuum_profile_id",
        "continuum_provider_id",
        "coupling_id",
        "geometry_sha256",
        "receiver_space",
        "scalar_id",
        "source_space",
        "state_equation_id",
        "topology_sha256",
    )

    capabilities = ()
    permanent_source_space = ATOMIC_L1_SOURCE_SPACE
    induced_source_space = ATOMIC_L1_SOURCE_SPACE
    linear_response = True
    structurally_rotation_equivariant = True
    model_field_is_energy_gradient = False
    general_source_primal_adjoint = True

    def __init__(
        self,
        *,
        atom_count: int,
        geometry_digest: str,
        positions_angstrom: object,
        continuum_provider_id: str,
        continuum_profile_id: str,
        continuum_configuration_sha256: str,
        continuum_provenance_sha256: str,
        topology_sha256: str,
        coordinate_kernel: HarmonicDDPCMHybridCoordinateKernel,
        schwarz_operator: object,
        conductor_operator: object,
        dielectric_operator: object,
        point_local_source: object,
        point_energy_receiver: object,
        gaussian_native_local_source: object,
        gaussian_induced_local_source: object,
    ) -> None:
        if type(atom_count) is not int or atom_count < 1:
            raise ValueError("atom_count must be a positive integer.")
        if not isinstance(coordinate_kernel, HarmonicDDPCMHybridCoordinateKernel):
            raise TypeError("coordinate_kernel has the wrong type.")
        positions = np.asarray(positions_angstrom, dtype=float)
        if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                f"positions_angstrom must be finite with shape ({atom_count}, 3)."
            )
        source_dimension = atom_count * 4
        receiver_dimension = atom_count * 8
        schwarz_array = np.asarray(schwarz_operator, dtype=float)
        if schwarz_array.ndim != 2 or schwarz_array.shape[0] != schwarz_array.shape[1]:
            raise ValueError("Schwarz operator must be square.")
        surface_dimension = schwarz_array.shape[0]
        square = (surface_dimension, surface_dimension)
        schwarz = _readonly_matrix(
            schwarz_array, shape=square, name="Schwarz operator"
        )
        conductor = _readonly_matrix(
            conductor_operator, shape=square, name="conductor operator"
        )
        dielectric = _readonly_matrix(
            dielectric_operator, shape=square, name="dielectric operator"
        )
        point_local = _readonly_matrix(
            point_local_source,
            shape=(surface_dimension, source_dimension),
            name="point local-source operator",
        )
        receiver = _readonly_matrix(
            point_energy_receiver,
            shape=(source_dimension, surface_dimension),
            name="point energy receiver",
        )
        gaussian_native = _readonly_matrix(
            gaussian_native_local_source,
            shape=(surface_dimension, receiver_dimension),
            name="Gaussian native local-source operator",
        )
        gaussian_induced = _readonly_matrix(
            gaussian_induced_local_source,
            shape=(surface_dimension, source_dimension),
            name="Gaussian induced local-source operator",
        )
        for name, matrix in (("Schwarz", schwarz), ("dielectric", dielectric)):
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            if singular_values[-1] <= 1.0e-12 or not np.isfinite(
                singular_values[0] / singular_values[-1]
            ):
                raise ValueError(f"{name} operator failed its invertibility gate.")

        common = np.linalg.solve(
            schwarz,
            np.linalg.solve(dielectric, conductor),
        )
        permanent_reaction = -(common @ point_local)
        induced_reaction = -(common @ gaussian_induced)
        adjoint_total = common.T @ receiver.T
        native_field_total = -(gaussian_native.T @ adjoint_total)

        reaction_combined = np.concatenate(
            (permanent_reaction, induced_reaction), axis=1
        )
        psi_receiver = receiver @ reaction_combined
        raw_hessian = np.vstack((psi_receiver, psi_receiver))
        hessian_scale = max(1.0, float(np.linalg.norm(raw_hessian, ord="fro")))
        raw_asymmetry = float(
            np.linalg.norm(raw_hessian - raw_hessian.T, ord="fro")
            / hessian_scale
        )
        energy_hessian = 0.5 * (raw_hessian + raw_hessian.T)

        object.__setattr__(self, "atom_count", atom_count)
        object.__setattr__(self, "source_space", ATOMIC_L1_SOURCE_SPACE)
        object.__setattr__(
            self, "receiver_space", MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
        )
        object.__setattr__(self, "coupling_id", HARMONIC_DDPCM_HYBRID_COUPLING_ID)
        object.__setattr__(
            self,
            "scalar_id",
            OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2,
        )
        object.__setattr__(self, "state_equation_id", "")
        object.__setattr__(self, "geometry_sha256", geometry_digest)
        object.__setattr__(self, "topology_sha256", topology_sha256)
        object.__setattr__(self, "cavity_profile_id", HARMONIC_DDPCM_HYBRID_CAVITY_PROFILE_ID)
        object.__setattr__(self, "continuum_provider_id", continuum_provider_id)
        object.__setattr__(self, "continuum_profile_id", continuum_profile_id)
        object.__setattr__(
            self,
            "continuum_configuration_sha256",
            continuum_configuration_sha256,
        )
        object.__setattr__(
            self,
            "_positions_angstrom",
            tuple(tuple(float(value) for value in row) for row in positions),
        )
        object.__setattr__(self, "_coordinate_kernel", coordinate_kernel)
        object.__setattr__(self, "_forward_boundary_dimension", surface_dimension)
        for name, matrix in (
            ("_surface_schwarz_values", schwarz),
            ("_surface_conductor_values", conductor),
            ("_surface_dielectric_values", dielectric),
            ("_point_local_values", point_local),
            ("_energy_receiver_values", receiver),
            ("_gaussian_native_local_values", gaussian_native),
            ("_gaussian_induced_local_values", gaussian_induced),
            ("_permanent_reaction_map_values", permanent_reaction),
            ("_induced_reaction_map_values", induced_reaction),
            ("_adjoint_total_map_values", adjoint_total),
            ("_native_field_total_map_values", native_field_total),
            ("_combined_hessian_values", energy_hessian),
        ):
            object.__setattr__(self, name, tuple(float(value) for value in matrix.flat))
        object.__setattr__(self, "_raw_hessian_asymmetry", raw_asymmetry)
        configuration = self._current_configuration_sha256()
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "_provenance_sha256",
            _canonical_sha256(
                {
                    "contract": HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
                    "configuration_sha256": configuration,
                    "continuum_provenance_sha256": continuum_provenance_sha256,
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
            raise AttributeError("HarmonicDDPCMHybridSnapshot is immutable.")
        object.__setattr__(self, name, value)

    @property
    def forward_boundary_dimension(self) -> int:
        return self._forward_boundary_dimension

    @property
    def boundary_dimension(self) -> int:
        """Dimension of the stored complete ``(X,xi)`` general-source state."""

        return 2 * self.forward_boundary_dimension

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
    def general_source_raw_asymmetry_relative_defect(self) -> float:
        """Return the expected raw ``psi/phi`` quadratic asymmetry.

        This is diagnostic only.  A general source need not make the raw
        bilinear matrix symmetric; the scalar uses its exact symmetric part.
        """

        return self._raw_hessian_asymmetry

    @property
    def permanent_field_map(self) -> np.ndarray:
        return _readonly_matrix(
            self._native_field_total_map_values,
            shape=(self.atom_count * 8, self.atom_count * 4),
            name="permanent field map",
        )

    @property
    def induced_field_map(self) -> np.ndarray:
        return self.permanent_field_map

    @property
    def combined_energy_hessian(self) -> np.ndarray:
        dimension = self.atom_count * 8
        return _readonly_matrix(
            self._combined_hessian_values,
            shape=(dimension, dimension),
            name="combined energy Hessian",
        )

    def _matrix(
        self, values: tuple[float, ...], shape: tuple[int, int], name: str
    ) -> np.ndarray:
        return _readonly_matrix(values, shape=shape, name=name)

    @property
    def permanent_reaction_map(self) -> np.ndarray:
        return self._matrix(
            self._permanent_reaction_map_values,
            (self.forward_boundary_dimension, self.atom_count * 4),
            "permanent reaction map",
        )

    @property
    def induced_reaction_map(self) -> np.ndarray:
        return self._matrix(
            self._induced_reaction_map_values,
            (self.forward_boundary_dimension, self.atom_count * 4),
            "induced reaction map",
        )

    @property
    def adjoint_total_map(self) -> np.ndarray:
        return self._matrix(
            self._adjoint_total_map_values,
            (self.forward_boundary_dimension, self.atom_count * 4),
            "adjoint total-source map",
        )

    @property
    def gaussian_native_local_source(self) -> np.ndarray:
        return self._matrix(
            self._gaussian_native_local_values,
            (self.forward_boundary_dimension, self.atom_count * 8),
            "Gaussian native local-source operator",
        )

    def _current_configuration_sha256(self) -> str:
        matrices = {
            "schwarz": self._surface_schwarz_values,
            "conductor": self._surface_conductor_values,
            "dielectric": self._surface_dielectric_values,
            "point_local": self._point_local_values,
            "point_receiver": self._energy_receiver_values,
            "gaussian_native_local": self._gaussian_native_local_values,
            "gaussian_induced_local": self._gaussian_induced_local_values,
            "permanent_reaction": self._permanent_reaction_map_values,
            "induced_reaction": self._induced_reaction_map_values,
            "adjoint_total": self._adjoint_total_map_values,
            "native_field_total": self._native_field_total_map_values,
            "energy_hessian": self._combined_hessian_values,
        }
        return _canonical_sha256(
            {
                "contract": HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
                "coupling_id": self.coupling_id,
                "scalar_id": self.scalar_id,
                "geometry_sha256": self.geometry_sha256,
                "topology_sha256": self.topology_sha256,
                "continuum_provider_id": self.continuum_provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "continuum_configuration_sha256": (
                    self.continuum_configuration_sha256
                ),
                "coordinate_kernel_sha256": (
                    self._coordinate_kernel.configuration_sha256()
                ),
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "positions_angstrom_sha256": _matrix_sha256(
                    self.positions_angstrom
                ),
                "forward_boundary_dimension": self.forward_boundary_dimension,
                "stored_boundary_state": "concatenated-primal-X-and-adjoint-xi",
                "general_source_raw_asymmetry_relative_defect": (
                    self._raw_hessian_asymmetry
                ),
                "matrix_sha256": {
                    name: _matrix_sha256(values) for name, values in matrices.items()
                },
                "state_equations": (
                    "X=-L^-1 Aeps^-1 Ainf phi; xi=N.T Cpoint.T psi"
                ),
                "continuum_energy": "G=1/2 psi.T Cpoint X",
                "native_field": "u=-Bnative.T xi",
                "source_receiver_duality_assumed": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("hybrid harmonic ddPCM configuration drifted.")
        return current

    def _source(self, values: object, *, name: str) -> np.ndarray:
        return ATOMIC_L1_SOURCE_SPACE.validate(
            values, atom_count=self.atom_count, name=name
        )

    def solve_boundary(
        self, permanent_source: object, induced_source: object
    ) -> np.ndarray:
        permanent = self._source(
            permanent_source, name="permanent source"
        ).reshape(-1)
        induced = self._source(induced_source, name="induced source").reshape(-1)
        reaction = (
            self.permanent_reaction_map @ permanent
            + self.induced_reaction_map @ induced
        )
        adjoint = self.adjoint_total_map @ (permanent + induced)
        result = np.concatenate((reaction, adjoint))
        result.setflags(write=False)
        return result

    def native_field_from_boundary(self, boundary_state: object) -> np.ndarray:
        state = np.asarray(boundary_state, dtype=float)
        if state.shape != (self.boundary_dimension,) or not np.all(
            np.isfinite(state)
        ):
            raise ValueError(
                "general-source boundary state must contain finite (X,xi)."
            )
        adjoint = state[self.forward_boundary_dimension :]
        field = -(self.gaussian_native_local_source.T @ adjoint)
        return self.receiver_space.validate(
            field.reshape(self.atom_count, 8),
            atom_count=self.atom_count,
            name="hybrid general-source native field",
        )

    def native_field(
        self, permanent_source: object, induced_source: object
    ) -> np.ndarray:
        return self.native_field_from_boundary(
            self.solve_boundary(permanent_source, induced_source)
        )

    def induced_field_jvp(self, induced_direction: object) -> np.ndarray:
        direction = self._source(
            induced_direction, name="induced direction"
        ).reshape(-1)
        result = self.induced_field_map @ direction
        return self.receiver_space.validate(
            result.reshape(self.atom_count, 8),
            atom_count=self.atom_count,
            name="induced native-field JVP",
        )

    def model_field_vjps(
        self, field_cotangent: object
    ) -> tuple[np.ndarray, np.ndarray]:
        cotangent = self.receiver_space.validate(
            field_cotangent,
            atom_count=self.atom_count,
            name="native field cotangent",
        ).reshape(-1)
        source = (self.permanent_field_map.T @ cotangent).reshape(
            self.atom_count, 4
        )
        return source.copy(), source.copy()

    def continuum_energy_eV(
        self, permanent_source: object, induced_source: object
    ) -> float:
        permanent = self._source(
            permanent_source, name="permanent source"
        ).reshape(-1)
        induced = self._source(induced_source, name="induced source").reshape(-1)
        combined = np.concatenate((permanent, induced))
        result = 0.5 * float(
            combined @ (self.combined_energy_hessian @ combined)
        )
        if not np.isfinite(result):
            raise RuntimeError("hybrid ddPCM continuum energy is non-finite.")
        return result

    def continuum_source_gradients(
        self, permanent_source: object, induced_source: object
    ) -> tuple[np.ndarray, np.ndarray]:
        permanent = self._source(
            permanent_source, name="permanent source"
        ).reshape(-1)
        induced = self._source(induced_source, name="induced source").reshape(-1)
        gradient = self.combined_energy_hessian @ np.concatenate(
            (permanent, induced)
        )
        split = self.atom_count * 4
        return (
            gradient[:split].reshape(self.atom_count, 4),
            gradient[split:].reshape(self.atom_count, 4),
        )

    def continuum_energy_position_gradient(
        self, permanent_source: object, induced_source: object
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._coordinate_kernel.continuum_energy_position_gradient(
            self.positions_angstrom,
            self._source(permanent_source, name="permanent source"),
            self._source(induced_source, name="induced source"),
        )

    def model_field_position_vjp(
        self,
        permanent_source: object,
        induced_source: object,
        field_cotangent: object,
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._coordinate_kernel.model_field_position_vjp(
            self.positions_angstrom,
            self._source(permanent_source, name="permanent source"),
            self._source(induced_source, name="induced source"),
            self.receiver_space.validate(
                field_cotangent,
                atom_count=self.atom_count,
                name="native field cotangent",
            ),
        )


def build_harmonic_ddpcm_hybrid_snapshot(
    continuum: SmoothPartitionHarmonicDDPCMFunctionalCandidate,
    geometry: object,
    *,
    receiver_radial_quadrature_order: int = 128,
) -> HarmonicDDPCMHybridSnapshot:
    """Build one geometry-bound heterogeneous general-source snapshot."""

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
    # The receiver order is content-addressed for profile continuity.  The
    # general-source receiver itself uses the same localized Gaussian MEP
    # assembly as the forward phi operator and therefore the scalar source
    # quadrature order, not a separate full-sphere charge reconstruction.
    kernel = HarmonicDDPCMHybridCoordinateKernel.from_functional(
        continuum,
        receiver_radial_quadrature_order=receiver_radial_quadrature_order,
    )
    matrices = kernel.matrices(positions)
    scalar_matrices = continuum.debug_geometry_matrices(geometry)
    mapping = {
        "schwarz_operator": "schwarz_operator",
        "conductor_operator": "conductor_operator",
        "dielectric_operator": "dielectric_operator",
        "point_local_source": "source_operator",
        "point_energy_receiver": "receiver",
    }
    for kernel_name, scalar_name in mapping.items():
        if not np.allclose(
            matrices[kernel_name],
            scalar_matrices[scalar_name],
            rtol=0.0,
            atol=3.0e-12,
        ):
            raise RuntimeError(
                f"coordinate-kernel {kernel_name} does not replay the scalar graph."
            )
    topology_sha256 = _canonical_sha256(
        {
            "contract": HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
            "continuum_configuration_sha256": continuum.configuration_sha256(),
            "atom_count": atom_count,
            "surface_lmax": continuum.surface_lmax,
            "coefficient_order": (
                "atom-major/l-ascending/m-ascending/real-harmonic"
            ),
            "forward_coefficient_dimension": int(
                matrices["schwarz_operator"].shape[0]
            ),
            "stored_state": "primal-plus-adjoint-fixed-dimension",
            "active_set": "none-fixed-coefficient-topology",
        }
    )
    return HarmonicDDPCMHybridSnapshot(
        atom_count=atom_count,
        geometry_digest=geometry_sha256(geometry),
        positions_angstrom=positions,
        continuum_provider_id=continuum.provider_id,
        continuum_profile_id=continuum.continuum_profile_id,
        continuum_configuration_sha256=continuum.configuration_sha256(),
        continuum_provenance_sha256=continuum.provenance_sha256,
        topology_sha256=topology_sha256,
        coordinate_kernel=kernel,
        schwarz_operator=matrices["schwarz_operator"],
        conductor_operator=matrices["conductor_operator"],
        dielectric_operator=matrices["dielectric_operator"],
        point_local_source=matrices["point_local_source"],
        point_energy_receiver=matrices["point_energy_receiver"],
        gaussian_native_local_source=matrices["gaussian_native_local_source"],
        gaussian_induced_local_source=matrices["gaussian_induced_local_source"],
    )


__all__ = [
    "HARMONIC_DDPCM_HYBRID_CAVITY_PROFILE_ID",
    "HARMONIC_DDPCM_HYBRID_CONTRACT_ID",
    "HARMONIC_DDPCM_HYBRID_COUPLING_ID",
    "HARMONIC_DDPCM_HYBRID_PROFILE_ID",
    "HarmonicDDPCMHybridCoordinateKernel",
    "HarmonicDDPCMHybridSnapshot",
    "build_harmonic_ddpcm_hybrid_snapshot",
]

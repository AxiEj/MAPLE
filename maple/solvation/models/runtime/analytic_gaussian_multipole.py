"""Analytic molecular real-space Gaussian multipole evaluator for MACE-POLAR.

This module is a separately identified inference candidate.  It replaces the
fixed laboratory-axis finite-difference stencils used by
``graph_longrange==0.4.0`` with the exact value, gradient, and Hessian of the
isotropic Gaussian-smoothed Coulomb kernel.  It does not change checkpoint
weights and it does not admit any Route-2 capability by itself.

The raw real-spherical ``l=1`` order used by the pinned checkpoint is
``(y, z, x)``.  Cartesian conversion occurs only at the two explicit indexing
sites below.  All other tensor algebra is Cartesian and therefore contains no
preferred laboratory direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

from importlib.metadata import PackageNotFoundError, version

import torch

from maple.function.calculator.mace._macepol_long_range import (
    MACEPolarLongRangeEvaluator,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)

_VALIDATED_GRAPH_LONGRANGE_VERSION = "0.4.0"
_FIELD_CONSTANT = 1.0 / (5.526349406e-3)
_COULOMB_PREFACTOR = _FIELD_CONSTANT / (4.0 * math.pi)
_DEFAULT_MINIMUM_SEPARATION_ANGSTROM = 1.0e-10
_EXPECTED_SOURCE_SIGMA_ANGSTROM = 1.5
_EXPECTED_RECEIVER_SIGMAS_ANGSTROM = (1.5, 3.0)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor hash input must be a Torch tensor.")
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _module_descriptor(module: torch.nn.Module) -> dict[str, object]:
    try:
        raw_path = inspect.getsourcefile(type(module))
    except (TypeError, OSError):
        raw_path = None
    if raw_path is None:
        raise RuntimeError(
            f"Source file is unavailable for {type(module).__qualname__}."
        )
    source_path = Path(raw_path).resolve()
    if not source_path.is_file():
        raise RuntimeError(
            f"Source path is unavailable for {type(module).__qualname__}."
        )
    return {
        "module": type(module).__module__,
        "qualname": type(module).__qualname__,
        "source_sha256": _file_sha256(source_path),
        "state": {
            name: _tensor_sha256(tensor)
            for name, tensor in sorted(module.state_dict().items())
        },
        "num_radial": getattr(module, "num_radial", None),
        "non_zero_terms": getattr(module, "non_zero_terms", None),
    }


def _source_matrix(source_feats: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(source_feats) or not torch.is_floating_point(source_feats):
        raise TypeError("source_feats must be one floating-point Torch tensor.")
    if source_feats.ndim == 3 and source_feats.shape[-2:] == (1, 4):
        return source_feats[:, 0, :]
    if source_feats.ndim == 2 and source_feats.shape[-1] == 4:
        return source_feats
    raise ValueError("source_feats must have shape (N,4) or (N,1,4).")


def _validate_geometry(
    source: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
) -> None:
    atom_count = source.shape[0]
    if (
        not torch.is_tensor(positions)
        or not torch.is_floating_point(positions)
        or positions.shape != (atom_count, 3)
    ):
        raise ValueError("positions must be floating point with shape (N,3).")
    if source.dtype != positions.dtype or source.device != positions.device:
        raise ValueError("source_feats and positions must share dtype and device.")
    if not bool(torch.isfinite(source).all()) or not bool(
        torch.isfinite(positions).all()
    ):
        raise ValueError("source_feats and positions must be finite.")
    if (
        not torch.is_tensor(batch)
        or batch.shape != (atom_count,)
        or batch.device != positions.device
        or batch.dtype == torch.bool
        or torch.is_floating_point(batch)
        or torch.is_complex(batch)
    ):
        raise ValueError("batch must be one integer Torch tensor with shape (N,).")
    if atom_count == 0 or bool(torch.any(batch < 0)):
        raise ValueError(
            "batch must describe at least one atom and use nonnegative IDs."
        )
    graph_count = int(batch.max().detach().cpu()) + 1
    expected = torch.arange(graph_count, dtype=batch.dtype, device=batch.device)
    if not torch.equal(torch.unique(batch, sorted=True), expected):
        raise ValueError("batch graph IDs must be contiguous from zero.")


def _pairwise_kernel_jet(
    positions: torch.Tensor,
    batch: torch.Tensor,
    total_width_factors: torch.Tensor,
    *,
    minimum_separation_angstrom: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``K``, ``grad_d K`` and ``Hess_d K`` for ``d=R_s-R_r``."""

    if (
        total_width_factors.ndim != 1
        or total_width_factors.numel() == 0
        or total_width_factors.dtype != positions.dtype
        or total_width_factors.device != positions.device
        or not bool(torch.isfinite(total_width_factors).all())
        or bool(torch.any(total_width_factors <= 0.0))
    ):
        raise ValueError(
            "total_width_factors must be finite positive values on the model dtype/device."
        )
    displacement = positions[:, None, :] - positions[None, :, :]
    same_graph = batch[:, None] == batch[None, :]
    off_diagonal = ~torch.eye(
        positions.shape[0], dtype=torch.bool, device=positions.device
    )
    pair_mask = same_graph & off_diagonal
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    pair_distances = distance[pair_mask]
    if pair_distances.numel() and bool(
        torch.any(pair_distances <= minimum_separation_angstrom)
    ):
        raise ValueError(
            "Distinct atoms in one graph are coincident or below the analytic "
            "Gaussian multipole separation guard."
        )

    safe_distance = torch.where(pair_mask, distance, torch.ones_like(distance))
    inverse_distance = safe_distance.reciprocal()
    unit = displacement * inverse_distance[..., None]
    width = total_width_factors.reshape(1, 1, -1)
    a = 0.5 / width
    ar = a * safe_distance[..., None]
    gaussian = torch.exp(-(ar * ar))
    erf = torch.erf(ar)
    c = 2.0 * a / math.sqrt(math.pi)

    inverse_r = inverse_distance[..., None]
    kernel_radial = erf * inverse_r
    first_radial = c * gaussian * inverse_r - erf * inverse_r.square()
    second_radial = (
        -2.0 * a.square() * c * gaussian
        - 2.0 * c * gaussian * inverse_r.square()
        + 2.0 * erf * inverse_r.pow(3)
    )

    kernel = _COULOMB_PREFACTOR * kernel_radial
    gradient = _COULOMB_PREFACTOR * first_radial[..., None] * unit[:, :, None, :]
    radial_over_r = first_radial * inverse_r
    outer = unit[..., :, None] * unit[..., None, :]
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    hessian = _COULOMB_PREFACTOR * (
        (second_radial - radial_over_r)[..., None, None] * outer[:, :, None, :, :]
        + radial_over_r[..., None, None] * identity
    )

    mask = pair_mask.to(dtype=positions.dtype)
    kernel = kernel * mask[..., None]
    gradient = gradient * mask[..., None, None]
    hessian = hessian * mask[..., None, None, None]
    return kernel, gradient, hessian, pair_mask


class AnalyticGaussianMultipoleElectrostaticFeatures(torch.nn.Module):
    """Exact molecular Gaussian ``l<=1`` source-to-receiver features."""

    def __init__(
        self,
        *,
        density_smearing_width: float,
        projection_smearing_widths: tuple[float, ...],
        include_self_interaction: bool,
        total_width_factors: torch.Tensor,
        l0_factors: torch.Tensor,
        l1_limit_factors: torch.Tensor,
        self_interaction: torch.nn.Module,
        minimum_separation_angstrom: float,
    ) -> None:
        super().__init__()
        self.density_max_l = 1
        self.projection_max_l = 1
        self.density_smearing_width = float(density_smearing_width)
        self.projection_smearing_widths = tuple(
            float(value) for value in projection_smearing_widths
        )
        self.num_radial = len(self.projection_smearing_widths)
        self.include_self_interaction = bool(include_self_interaction)
        self.minimum_separation_angstrom = float(minimum_separation_angstrom)
        self.self_interaction = self_interaction
        self.register_buffer(
            "total_width_factors", total_width_factors.detach().clone()
        )
        self.register_buffer("l0_factors", l0_factors.detach().clone())
        self.register_buffer("l1_limit_factors", l1_limit_factors.detach().clone())
        self._validate_configuration()

    @classmethod
    def from_upstream(
        cls,
        upstream: torch.nn.Module,
        *,
        minimum_separation_angstrom: float,
    ) -> "AnalyticGaussianMultipoleElectrostaticFeatures":
        required = (
            "density_max_l",
            "projection_max_l",
            "density_smearing_width",
            "projection_smearing_widths",
            "num_radial",
            "include_self_interaction",
            "total_width_factors",
            "l0_factors",
            "l1_factors",
            "offset",
            "self_interaction",
        )
        missing = [name for name in required if not hasattr(upstream, name)]
        if missing:
            raise RuntimeError(
                "The upstream real-space feature module is missing: "
                + ", ".join(missing)
                + "."
            )
        if int(upstream.density_max_l) != 1 or int(upstream.projection_max_l) != 1:
            raise RuntimeError(
                "The analytic evaluator supports only the pinned l<=1 path."
            )
        offset = float(upstream.offset)
        if not math.isfinite(offset) or offset <= 0.0:
            raise RuntimeError("The upstream feature stencil offset is invalid.")
        return cls(
            density_smearing_width=float(upstream.density_smearing_width),
            projection_smearing_widths=tuple(
                float(value) for value in upstream.projection_smearing_widths
            ),
            include_self_interaction=bool(upstream.include_self_interaction),
            total_width_factors=upstream.total_width_factors,
            l0_factors=upstream.l0_factors,
            l1_limit_factors=upstream.l1_factors * offset,
            self_interaction=upstream.self_interaction,
            minimum_separation_angstrom=minimum_separation_angstrom,
        )

    def _validate_configuration(self) -> None:
        if self.density_smearing_width != _EXPECTED_SOURCE_SIGMA_ANGSTROM:
            raise ValueError("Unexpected checkpoint density smearing width.")
        if self.projection_smearing_widths != _EXPECTED_RECEIVER_SIGMAS_ANGSTROM:
            raise ValueError("Unexpected checkpoint receiver smearing widths.")
        if self.num_radial != 2:
            raise ValueError(
                "The pinned analytic feature profile requires two radial channels."
            )
        for name in ("total_width_factors", "l0_factors", "l1_limit_factors"):
            tensor = getattr(self, name)
            if (
                tensor.shape != (self.num_radial,)
                or not torch.is_floating_point(tensor)
                or not bool(torch.isfinite(tensor).all())
                or bool(torch.any(tensor <= 0.0))
            ):
                raise ValueError(f"{name} must contain two finite positive values.")
        if (
            not math.isfinite(self.minimum_separation_angstrom)
            or self.minimum_separation_angstrom <= 0.0
        ):
            raise ValueError("minimum_separation_angstrom must be finite and positive.")
        if not isinstance(self.self_interaction, torch.nn.Module):
            raise TypeError("self_interaction must be one Torch module.")

    def configuration_sha256(self) -> str:
        self._validate_configuration()
        return _canonical_sha256(
            {
                "schema": "route2-analytic-gaussian-multipole-features-v1",
                "density_max_l": self.density_max_l,
                "projection_max_l": self.projection_max_l,
                "density_smearing_width": self.density_smearing_width,
                "projection_smearing_widths": self.projection_smearing_widths,
                "include_self_interaction": self.include_self_interaction,
                "minimum_separation_angstrom": self.minimum_separation_angstrom,
                "field_constant": _FIELD_CONSTANT,
                "total_width_factors_sha256": _tensor_sha256(self.total_width_factors),
                "l0_factors_sha256": _tensor_sha256(self.l0_factors),
                "l1_limit_factors_sha256": _tensor_sha256(self.l1_limit_factors),
                "self_interaction": _module_descriptor(self.self_interaction),
            }
        )

    def forward(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        self._validate_configuration()
        source = _source_matrix(source_feats)
        _validate_geometry(source, node_positions, batch)
        for tensor in (
            self.total_width_factors,
            self.l0_factors,
            self.l1_limit_factors,
        ):
            if tensor.dtype != source.dtype or tensor.device != source.device:
                raise ValueError(
                    "analytic feature buffers drifted from model dtype/device."
                )
        kernel, gradient, hessian, _ = _pairwise_kernel_jet(
            node_positions,
            batch,
            self.total_width_factors,
            minimum_separation_angstrom=self.minimum_separation_angstrom,
        )
        charge = source[:, 0]
        # Checkpoint raw real-spherical order is (y,z,x).
        dipole_cartesian = source[:, (3, 1, 2)]
        scalar = charge[:, None, None] * kernel + torch.einsum(
            "sa,srka->srk", dipole_cartesian, gradient
        )
        l0 = torch.sum(scalar, dim=0) * self.l0_factors

        source_gradient = charge[:, None, None, None] * gradient + torch.einsum(
            "srkab,sa->srkb", hessian, dipole_cartesian
        )
        receiver_cartesian = -torch.sum(source_gradient, dim=0)
        receiver_raw = receiver_cartesian[..., (1, 2, 0)]
        l1 = receiver_raw * self.l1_limit_factors[None, :, None]
        features = torch.cat((l0, l1.reshape(source.shape[0], -1)), dim=-1)

        self_terms = self.self_interaction(source)
        if self_terms.shape != features.shape or not bool(
            torch.isfinite(self_terms).all()
        ):
            raise RuntimeError(
                "Analytic feature self interaction must be finite with shape (N,8)."
            )
        if self.include_self_interaction:
            features = features + self_terms
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("Analytic Gaussian multipole features are non-finite.")
        return features, self_terms, None


class AnalyticGaussianMultipoleElectrostaticEnergy(torch.nn.Module):
    """Exact molecular Gaussian ``l<=1`` pair energy."""

    def __init__(
        self,
        *,
        density_smearing_width: float,
        include_self_interaction: bool,
        self_interaction: torch.nn.Module,
        minimum_separation_angstrom: float,
    ) -> None:
        super().__init__()
        self.density_max_l = 1
        self.density_smearing_width = float(density_smearing_width)
        self.include_self_interaction = bool(include_self_interaction)
        self.minimum_separation_angstrom = float(minimum_separation_angstrom)
        self.self_interaction = self_interaction
        self.register_buffer(
            "total_width_factor",
            torch.as_tensor(
                [self.density_smearing_width], dtype=torch.get_default_dtype()
            ),
        )
        self._validate_configuration()

    @classmethod
    def from_upstream(
        cls,
        upstream: torch.nn.Module,
        *,
        minimum_separation_angstrom: float,
    ) -> "AnalyticGaussianMultipoleElectrostaticEnergy":
        required = (
            "density_max_l",
            "density_smearing_width",
            "include_self_interaction",
            "self_interaction",
        )
        missing = [name for name in required if not hasattr(upstream, name)]
        if missing:
            raise RuntimeError(
                "The upstream real-space energy module is missing: "
                + ", ".join(missing)
                + "."
            )
        if int(upstream.density_max_l) != 1:
            raise RuntimeError(
                "The analytic evaluator supports only the pinned l<=1 path."
            )
        result = cls(
            density_smearing_width=float(upstream.density_smearing_width),
            include_self_interaction=bool(upstream.include_self_interaction),
            self_interaction=upstream.self_interaction,
            minimum_separation_angstrom=minimum_separation_angstrom,
        )
        try:
            template = next(upstream.self_interaction.buffers())
        except StopIteration as exc:
            raise RuntimeError(
                "The upstream energy self-interaction module has no bound tensors."
            ) from exc
        result.to(dtype=template.dtype, device=template.device)
        return result

    def _validate_configuration(self) -> None:
        if self.density_smearing_width != _EXPECTED_SOURCE_SIGMA_ANGSTROM:
            raise ValueError("Unexpected checkpoint density smearing width.")
        if (
            self.total_width_factor.shape != (1,)
            or not torch.is_floating_point(self.total_width_factor)
            or not bool(torch.isfinite(self.total_width_factor).all())
            or bool(torch.any(self.total_width_factor <= 0.0))
            or float(self.total_width_factor.detach().cpu()[0])
            != self.density_smearing_width
        ):
            raise ValueError("The analytic energy width buffer is inconsistent.")
        if (
            not math.isfinite(self.minimum_separation_angstrom)
            or self.minimum_separation_angstrom <= 0.0
        ):
            raise ValueError("minimum_separation_angstrom must be finite and positive.")
        if not isinstance(self.self_interaction, torch.nn.Module):
            raise TypeError("self_interaction must be one Torch module.")

    def configuration_sha256(self) -> str:
        self._validate_configuration()
        return _canonical_sha256(
            {
                "schema": "route2-analytic-gaussian-multipole-energy-v1",
                "density_max_l": self.density_max_l,
                "density_smearing_width": self.density_smearing_width,
                "include_self_interaction": self.include_self_interaction,
                "minimum_separation_angstrom": self.minimum_separation_angstrom,
                "field_constant": _FIELD_CONSTANT,
                "total_width_factor_sha256": _tensor_sha256(self.total_width_factor),
                "self_interaction": _module_descriptor(self.self_interaction),
            }
        )

    def forward(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_configuration()
        source = _source_matrix(source_feats)
        _validate_geometry(source, positions, batch)
        if (
            self.total_width_factor.dtype != source.dtype
            or self.total_width_factor.device != source.device
        ):
            raise ValueError("analytic energy width drifted from model dtype/device.")
        kernel, gradient, hessian, _ = _pairwise_kernel_jet(
            positions,
            batch,
            self.total_width_factor,
            minimum_separation_angstrom=self.minimum_separation_angstrom,
        )
        kernel = kernel[..., 0]
        gradient = gradient[..., 0, :]
        hessian = hessian[..., 0, :, :]
        charge = source[:, 0]
        dipole = source[:, (3, 1, 2)]

        pair_energy = charge[:, None] * charge[None, :] * kernel
        pair_energy = pair_energy + charge[None, :] * torch.einsum(
            "sa,sra->sr", dipole, gradient
        )
        pair_energy = pair_energy - charge[:, None] * torch.einsum(
            "ra,sra->sr", dipole, gradient
        )
        pair_energy = pair_energy - torch.einsum(
            "ra,srab,sb->sr", dipole, hessian, dipole
        )
        node_energy = 0.5 * torch.sum(pair_energy, dim=0)
        graph_count = int(batch.max().detach().cpu()) + 1
        result = torch.zeros(graph_count, dtype=source.dtype, device=source.device)
        result.index_add_(0, batch, node_energy)

        if self.include_self_interaction:
            self_fields = self.self_interaction(source)
            if self_fields.shape != source.shape or not bool(
                torch.isfinite(self_fields).all()
            ):
                raise RuntimeError(
                    "Analytic energy self interaction must be finite with shape (N,4)."
                )
            self_node_energy = torch.einsum("nb,nb->n", source, self_fields)
            self_energy = torch.zeros_like(result)
            self_energy.index_add_(0, batch, self_node_energy)
            result = result + 0.5 * self_energy
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Analytic Gaussian multipole energy is non-finite.")
        return result


@dataclass(frozen=True)
class MACEPolarAnalyticGaussianMultipoleEvaluator(MACEPolarLongRangeEvaluator):
    """One immutable molecular evaluator policy for the analytic candidate."""

    minimum_separation_angstrom: float = _DEFAULT_MINIMUM_SEPARATION_ANGSTROM
    _installed_configuration_sha256: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.profile != MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID:
            raise ValueError("Unsupported analytic Gaussian multipole profile.")
        if self.use_pbc_evaluator is not False or self.box_length_angstrom is not None:
            raise ValueError(
                "The analytic Gaussian multipole profile is molecular only."
            )
        if (
            not math.isfinite(self.minimum_separation_angstrom)
            or self.minimum_separation_angstrom <= 0.0
        ):
            raise ValueError("minimum_separation_angstrom must be finite and positive.")

    @classmethod
    def from_profile(
        cls, profile: str
    ) -> "MACEPolarAnalyticGaussianMultipoleEvaluator":
        normalized = str(profile).strip().lower()
        if normalized != MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID:
            raise ValueError(
                "Unsupported analytic Gaussian multipole evaluator profile: "
                f"{profile}."
            )
        return cls(normalized, False, None)

    @property
    def is_default(self) -> bool:
        # This retains the default molecular coordinate/batch policy.  It does
        # not claim numerical equivalence to the upstream fixed-axis operator.
        return True

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "operator": "analytic isotropic Gaussian l<=1 multipole real-space evaluator",
            "implementation_sha256": _file_sha256(Path(__file__).resolve()),
            "use_pbc_evaluator": False,
            "box_length_angstrom": None,
            "coordinate_policy": "upstream molecular batch; no body frame",
            "coordinate_derivative_policy": "identity pullback through scalar Torch graph",
            "pbc": False,
            "dtype_bridge": False,
            "graph_longrange_version": _VALIDATED_GRAPH_LONGRANGE_VERSION,
            "equivalent_to_default_evaluator": False,
            "checkpoint_weights_changed": False,
            "inference_operator_changed": True,
            "minimum_separation_angstrom": self.minimum_separation_angstrom,
            "installed_configuration_sha256": (self._installed_configuration_sha256),
            "structural_so3_contract": (
                "isotropic radial kernel plus Cartesian gradient/Hessian tensor contractions"
            ),
            "experimental": True,
            "capabilities": {tier: False for tier in "EFHVM"},
        }

    def configure_model(self, model: object) -> None:
        try:
            installed_version = version("graph-longrange")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "The analytic MACE-POLAR profile requires graph_longrange."
            ) from exc
        if installed_version != _VALIDATED_GRAPH_LONGRANGE_VERSION:
            raise RuntimeError(
                "The analytic MACE-POLAR profile is bound to graph_longrange "
                f"{_VALIDATED_GRAPH_LONGRANGE_VERSION}; found {installed_version}."
            )
        try:
            descriptor = model.electric_potential_descriptor
            energy = model.coulomb_energy
            old_features = descriptor.realspace_features
            old_energy = energy.realspace_energy
        except AttributeError as exc:
            raise RuntimeError(
                "The loaded checkpoint lacks the pinned molecular real-space modules."
            ) from exc
        if isinstance(old_features, AnalyticGaussianMultipoleElectrostaticFeatures):
            if not isinstance(old_energy, AnalyticGaussianMultipoleElectrostaticEnergy):
                raise RuntimeError("Only one analytic real-space module was installed.")
            current = self._current_configuration_sha256(model)
            if self._installed_configuration_sha256 is None:
                object.__setattr__(self, "_installed_configuration_sha256", current)
            self.configuration_sha256(model)
            return
        if isinstance(old_energy, AnalyticGaussianMultipoleElectrostaticEnergy):
            raise RuntimeError("Only one analytic real-space module was installed.")

        new_features = AnalyticGaussianMultipoleElectrostaticFeatures.from_upstream(
            old_features,
            minimum_separation_angstrom=self.minimum_separation_angstrom,
        )
        new_energy = AnalyticGaussianMultipoleElectrostaticEnergy.from_upstream(
            old_energy,
            minimum_separation_angstrom=self.minimum_separation_angstrom,
        )
        if new_features.include_self_interaction is not False:
            raise RuntimeError(
                "The pinned feature graph unexpectedly includes self terms."
            )
        if new_energy.include_self_interaction is not True:
            raise RuntimeError(
                "The pinned energy graph unexpectedly excludes self terms."
            )
        descriptor.realspace_features = new_features
        energy.realspace_energy = new_energy
        object.__setattr__(
            self,
            "_installed_configuration_sha256",
            self._current_configuration_sha256(model),
        )
        self.configuration_sha256(model)

    def _current_configuration_sha256(self, model: object) -> str:
        try:
            features = model.electric_potential_descriptor.realspace_features
            energy = model.coulomb_energy.realspace_energy
        except AttributeError as exc:
            raise RuntimeError(
                "The analytic evaluator model binding is incomplete."
            ) from exc
        if not isinstance(features, AnalyticGaussianMultipoleElectrostaticFeatures):
            raise RuntimeError(
                "The analytic Gaussian multipole feature module is absent."
            )
        if not isinstance(energy, AnalyticGaussianMultipoleElectrostaticEnergy):
            raise RuntimeError(
                "The analytic Gaussian multipole energy module is absent."
            )
        return _canonical_sha256(
            {
                "schema": "route2-mace-polar-analytic-gaussian-multipole-evaluator-v1",
                "profile": self.profile,
                "implementation_sha256": _file_sha256(Path(__file__).resolve()),
                "minimum_separation_angstrom": self.minimum_separation_angstrom,
                "features_configuration_sha256": features.configuration_sha256(),
                "energy_configuration_sha256": energy.configuration_sha256(),
            }
        )

    def configuration_sha256(self, model: object) -> str:
        current = self._current_configuration_sha256(model)
        expected = self._installed_configuration_sha256
        if expected is None:
            raise RuntimeError(
                "The analytic Gaussian multipole evaluator is not model-bound."
            )
        if current != expected:
            raise RuntimeError(
                "The analytic Gaussian multipole evaluator configuration drifted."
            )
        return current


__all__ = [
    "AnalyticGaussianMultipoleElectrostaticEnergy",
    "AnalyticGaussianMultipoleElectrostaticFeatures",
    "MACEPolarAnalyticGaussianMultipoleEvaluator",
]

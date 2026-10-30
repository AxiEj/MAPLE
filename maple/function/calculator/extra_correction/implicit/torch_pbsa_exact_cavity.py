"""Energy-only PBSA SAV cavity on the native half-Angstrom integer lattice.

This independent discrete counting kernel is not a solvent provider. It takes
already serialized native-evaluation coordinates and effective GAFF2 rmin
values; it does not prepare charges/topology or provide coordinate forces.
The lattice rule is backed by a frozen 30-molecule native component corpus
and separate coordinate-translation canaries, not by a translated Fortran
source body. Dispersion and CHA-GB/R6 remain separate components.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from torch import Tensor


GRID_SPACING_ANGSTROM = 0.5
CAVITY_PROBE_ANGSTROM = 1.3
CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 = 0.0378
CAVITY_OFFSET_KCAL_MOL = -0.5692
_MAX_PAIR_TEMPORARY_BYTES = 64 * 1024 * 1024
# Conservative displacement/square/reduction/mask temporaries per point/site.
_BYTES_PER_POINT_ATOM = 96
# Integer lattice coordinates must still be exactly representable in the selected dtype.
_MAX_EXACT_GRID_INDEX_FLOAT32 = 2**24 - 1
_MAX_EXACT_GRID_INDEX_FLOAT64 = 2**52 - 1


@dataclass(frozen=True)
class DiscretePbsaCavityResult:
    """One fixed-geometry voxel cavity contribution, never a differentiable PES."""

    energy_kcal_mol: Tensor
    occupied_voxels: int
    evaluated_grid_points: int
    supports_forces: ClassVar[bool] = False


def cavity_from_native_grid(
    positions_angstrom: Tensor,
    lj_rmin_angstrom: Tensor,
    *,
    max_grid_points: int = 4_000_000,
    max_pair_evaluations: int = 100_000_000,
    coordinates_validated_f12_7: bool = False,
    grid_chunk_size: int | None = None,
    readback_budget=None,
) -> DiscretePbsaCavityResult:
    """Count lattice points in the union of rmin+1.3-A atomic balls.

    The caller must supply the *serialized coordinates seen by Amber*, not
    higher-precision pre-serialization coordinates. A later public input
    boundary must perform that serialization once and pin its identity.
    """
    import torch

    if (
        not isinstance(positions_angstrom, torch.Tensor)
        or positions_angstrom.dtype not in (torch.float32, torch.float64)
        or positions_angstrom.ndim != 2
        or positions_angstrom.shape[1] != 3
        or positions_angstrom.shape[0] == 0
        or not isinstance(lj_rmin_angstrom, torch.Tensor)
        or lj_rmin_angstrom.dtype != positions_angstrom.dtype
        or lj_rmin_angstrom.shape != (len(positions_angstrom),)
        or lj_rmin_angstrom.device != positions_angstrom.device
    ):
        raise TypeError(
            "PBSA cavity requires same-device, same-dtype float32 or float64 "
            "[N,3] and [N] tensors."
        )
    if positions_angstrom.requires_grad or lj_rmin_angstrom.requires_grad:
        raise ValueError(
            "The discrete PBSA cavity is energy-only; forces are unsupported."
        )
    is_cuda = positions_angstrom.device.type == "cuda"
    private_cuda = is_cuda and positions_angstrom.dtype == torch.float32
    if private_cuda:
        if not coordinates_validated_f12_7:
            raise ValueError(
                "CUDA PBSA cavity requires F12.7 validation before upload."
            )
        if readback_budget is None:
            raise TypeError("CUDA PBSA cavity requires a readback budget.")
        input_error = (~torch.isfinite(positions_angstrom).all()).to(torch.int32)
        input_error |= (~torch.isfinite(lj_rmin_angstrom).all()).to(torch.int32) * 2
        input_error |= (lj_rmin_angstrom <= 0.0).any().to(torch.int32) * 4
        if readback_budget.read_count(
            input_error, stage="cavity", purpose="aggregate-input-validation"
        ):
            raise ValueError(
                "PBSA cavity coordinates and rmin must be finite and positive."
            )
    else:
        if not bool(
            torch.isfinite(positions_angstrom).all()
            and torch.isfinite(lj_rmin_angstrom).all()
            and (lj_rmin_angstrom > 0.0).all()
        ):
            raise ValueError(
                "PBSA cavity coordinates and rmin must be finite and positive."
            )
        for value in positions_angstrom.detach().cpu().reshape(-1).tolist():
            token = f"{value:12.7f}"
            # Python's field width is a minimum; an oversized token is not an
            # Amber F12.7 coordinate even when rounding leaves its value unchanged.
            serialized_value = float(token)
            if positions_angstrom.dtype == torch.float32:
                serialized_value = positions_angstrom.new_tensor(serialized_value).item()
            if len(token) != 12 or value != serialized_value:
                raise ValueError(
                    "PBSA cavity requires serialized native 12.7f coordinates."
                )
    if type(max_grid_points) is not int or max_grid_points < 1:
        raise ValueError("max_grid_points must be a positive integer.")
    if type(max_pair_evaluations) is not int or max_pair_evaluations < 1:
        raise ValueError("max_pair_evaluations must be a positive integer.")
    if grid_chunk_size is not None and (
        type(grid_chunk_size) is not int or grid_chunk_size < 1
    ):
        raise ValueError("grid_chunk_size must be None or a positive integer.")
    point_atom_bytes = len(positions_angstrom) * _BYTES_PER_POINT_ATOM
    if point_atom_bytes > _MAX_PAIR_TEMPORARY_BYTES:
        raise RuntimeError("PBSA cavity single-point pair memory budget exceeded.")

    if positions_angstrom.dtype == torch.float32:
        spacing = positions_angstrom.new_tensor(GRID_SPACING_ANGSTROM)
        probe = positions_angstrom.new_tensor(CAVITY_PROBE_ANGSTROM)
        expanded = lj_rmin_angstrom + probe
    else:
        spacing = GRID_SPACING_ANGSTROM
        expanded = lj_rmin_angstrom + CAVITY_PROBE_ANGSTROM
    lower = torch.floor(
        torch.min(positions_angstrom - expanded[:, None], dim=0).values / spacing
    )
    upper = torch.ceil(
        torch.max(positions_angstrom + expanded[:, None], dim=0).values / spacing
    )
    # Conversion of huge finite floats to int64 may yield the minimum integer
    # for BOTH bounds, falsely making a huge domain look like one grid point.
    # Reject before casting, not after the already-corrupted extent estimate.
    exact_limit = (
        _MAX_EXACT_GRID_INDEX_FLOAT32
        if positions_angstrom.dtype == torch.float32
        else _MAX_EXACT_GRID_INDEX_FLOAT64
    )
    bounds_valid = (
        torch.isfinite(lower).all()
        & torch.isfinite(upper).all()
        & (lower.abs() <= exact_limit).all()
        & (upper.abs() <= exact_limit).all()
    )
    if private_cuda:
        packed_bounds = torch.cat(
            (
                (~bounds_valid).to(torch.int64).reshape(1),
                lower.to(torch.int64),
                upper.to(torch.int64),
            )
        )
        host_bounds = readback_budget.read_scalars(
            packed_bounds,
            stage="cavity",
            purpose="grid-extents-validation",
        )
        if host_bounds[0]:
            raise RuntimeError("PBSA cavity grid index range exceeds exact arithmetic.")
        lower_host = tuple(int(value) for value in host_bounds[1:4])
        upper_host = tuple(int(value) for value in host_bounds[4:7])
    else:
        if not bool(bounds_valid):
            raise RuntimeError("PBSA cavity grid index range exceeds exact arithmetic.")
        lower_host = tuple(int(value) for value in lower.tolist())
        upper_host = tuple(int(value) for value in upper.tolist())
    lower, upper = lower.to(torch.int64), upper.to(torch.int64)
    extents = [high - low + 1 for low, high in zip(lower_host, upper_host)]
    if any(size <= 0 for size in extents):
        raise RuntimeError("PBSA cavity grid index range is empty or invalid.")
    grid_points = math.prod(extents)
    if grid_points > max_grid_points:
        raise RuntimeError(
            f"PBSA cavity grid budget exceeded: {grid_points} > {max_grid_points}."
        )
    pair_evaluations = grid_points * len(positions_angstrom)
    if pair_evaluations > max_pair_evaluations:
        raise RuntimeError(
            "PBSA cavity point-by-atom pair-work budget exceeded: "
            f"{pair_evaluations} > {max_pair_evaluations}."
        )

    count = torch.zeros((), dtype=torch.int64, device=positions_angstrom.device)
    yz_stride = extents[1] * extents[2]
    # The displacement and squared-distance intermediates scale with the
    # atom count as well as grid size. Bound each chunk instead of assuming
    # that a grid-point cap alone is a useful memory budget.
    chunk_size = min(
        32_768,
        _MAX_PAIR_TEMPORARY_BYTES // point_atom_bytes,
    )
    if grid_chunk_size is not None:
        chunk_size = min(chunk_size, grid_chunk_size)
    for start in range(0, grid_points, chunk_size):
        indices = torch.arange(
            start,
            min(start + chunk_size, grid_points),
            dtype=torch.int64,
            device=positions_angstrom.device,
        )
        xyz_indices = torch.stack(
            (
                indices // yz_stride,
                (indices // extents[2]) % extents[1],
                indices % extents[2],
            ),
            dim=1,
        )
        points = (xyz_indices + lower[None, :]).to(
            dtype=positions_angstrom.dtype
        ) * spacing
        squared_distances = (
            (points[:, None, :] - positions_angstrom[None, :, :]).square().sum(dim=-1)
        )
        count += (squared_distances <= expanded.square()[None, :]).any(dim=1).sum(
            dtype=torch.int64
        )

    if private_cuda:
        occupied_count = readback_budget.read_count(
            count, stage="cavity", purpose="final-occupied-voxel-count"
        )
    else:
        occupied_count = int(count)

    if positions_angstrom.dtype == torch.float64:
        energy = positions_angstrom.new_tensor(
            CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 * GRID_SPACING_ANGSTROM**3 * occupied_count
            + CAVITY_OFFSET_KCAL_MOL
        )
    else:
        coefficient = positions_angstrom.new_tensor(
            CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3
        )
        offset = positions_angstrom.new_tensor(CAVITY_OFFSET_KCAL_MOL)
        occupied = count.to(dtype=torch.float32)
        energy = coefficient * occupied * spacing.pow(3) + offset
    return DiscretePbsaCavityResult(energy, occupied_count, grid_points)

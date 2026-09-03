"""Torch-native COSMO surface and sigma-profile preprocessing.

The runtime objects in this module contain no ORCA or openCOSMO-RS process
calls.  ``parse_orca_cosmo`` is intentionally an offline interoperability
reader: it lets a published ORCA ``.orcacosmo`` surface act as a numerical
oracle while all averaging, gridding, and thermodynamics remain in PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from typing import Any

BOHR_ANGSTROM = 0.52917721092
OPEN_COSMORS_SIGMA_MIN = -0.15
OPEN_COSMORS_SIGMA_MAX = 0.15
OPEN_COSMORS_SIGMA_STEP = 0.001
OPEN_COSMORS_ORTHOGONAL_RADIUS_ANGSTROM = 1.0
OPEN_COSMORS_ORTHOGONAL_SUBTRACTION = 0.816


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise ImportError("Torch COSMO-RS preprocessing requires PyTorch.") from exc
    return torch


def _float64_tensor(value: object, *, name: str, device: object | None = None):
    torch = _torch()
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if device is not None and tensor.device != torch.device(device):
        tensor = tensor.to(device=device)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} must use torch.float64.")
    if not bool(torch.isfinite(tensor).all().detach()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _int64_tensor(value: object, *, name: str, device: object | None = None):
    torch = _torch()
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if device is not None and tensor.device != torch.device(device):
        tensor = tensor.to(device=device)
    if tensor.dtype != torch.int64:
        tensor = tensor.to(dtype=torch.int64)
    return tensor


@dataclass(frozen=True, slots=True)
class COSMOSurface:
    """One conductor surface with differentiable Torch segment tensors."""

    name: str
    atomic_numbers: tuple[int, ...]
    atom_positions_angstrom: Any
    segment_positions_angstrom: Any
    segment_areas_angstrom2: Any
    segment_parent_atom_indices: Any
    segment_screening_charge_e: Any
    dielectric_energy_hartree: Any
    dielectric_energy_role: str
    cavity_volume_angstrom3: Any
    molecular_charge_e: Any
    source_identity: str
    screening_charge_constraint: str

    def __post_init__(self) -> None:
        torch = _torch()
        name = str(self.name).strip()
        source_identity = str(self.source_identity).strip()
        numbers = tuple(self.atomic_numbers)
        if not name or not source_identity:
            raise ValueError(
                "COSMO surface name and source identity must be non-empty."
            )
        dielectric_role = str(self.dielectric_energy_role).strip()
        if dielectric_role not in {
            "total-solvated-minus-gas",
            "boundary-polarization-only",
        }:
            raise ValueError("COSMO dielectric_energy_role is unsupported.")
        charge_constraint = str(self.screening_charge_constraint).strip()
        if charge_constraint not in {"exact-total-charge", "file-as-provided"}:
            raise ValueError("COSMO screening_charge_constraint is unsupported.")
        if not numbers or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in numbers
        ):
            raise ValueError(
                "atomic_numbers must be a nonempty tuple of positive integers."
            )

        atoms = _float64_tensor(
            self.atom_positions_angstrom,
            name="atom_positions_angstrom",
        )
        device = atoms.device
        points = _float64_tensor(
            self.segment_positions_angstrom,
            name="segment_positions_angstrom",
            device=device,
        )
        areas = _float64_tensor(
            self.segment_areas_angstrom2,
            name="segment_areas_angstrom2",
            device=device,
        )
        parents = _int64_tensor(
            self.segment_parent_atom_indices,
            name="segment_parent_atom_indices",
            device=device,
        )
        charges = _float64_tensor(
            self.segment_screening_charge_e,
            name="segment_screening_charge_e",
            device=device,
        )
        dielectric = _float64_tensor(
            self.dielectric_energy_hartree,
            name="dielectric_energy_hartree",
            device=device,
        )
        volume = _float64_tensor(
            self.cavity_volume_angstrom3,
            name="cavity_volume_angstrom3",
            device=device,
        )
        molecular_charge = _float64_tensor(
            self.molecular_charge_e,
            name="molecular_charge_e",
            device=device,
        )
        segment_count = points.shape[0] if points.ndim == 2 else -1
        if atoms.shape != (len(numbers), 3):
            raise ValueError("atom_positions_angstrom has the wrong shape.")
        if points.ndim != 2 or points.shape[1] != 3 or segment_count == 0:
            raise ValueError("segment_positions_angstrom must have shape (S, 3).")
        if any(value.shape != (segment_count,) for value in (areas, parents, charges)):
            raise ValueError(
                "COSMO segment areas, parents, and charges must have shape (S,)."
            )
        if bool((areas <= 0.0).any().detach()):
            raise ValueError("COSMO segment areas must be positive.")
        if bool(((parents < 0) | (parents >= len(numbers))).any().detach()):
            raise ValueError("COSMO segment parents must index atomic_numbers.")
        if dielectric.ndim != 0 or volume.ndim != 0 or molecular_charge.ndim != 0:
            raise ValueError("COSMO scalar fields must be scalar tensors.")
        if float(volume.detach()) <= 0.0:
            raise ValueError("COSMO cavity volume must be positive.")
        charge_error = float(torch.abs(charges.sum() + molecular_charge).detach().cpu())
        charge_tolerance = 2.0e-8 * max(1.0, abs(float(molecular_charge.detach())))
        if (
            charge_constraint == "exact-total-charge"
            and charge_error > charge_tolerance
        ):
            raise ValueError(
                "Conductor screening charge must equal minus the molecular charge."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "atom_positions_angstrom", atoms)
        object.__setattr__(self, "segment_positions_angstrom", points)
        object.__setattr__(self, "segment_areas_angstrom2", areas)
        object.__setattr__(self, "segment_parent_atom_indices", parents)
        object.__setattr__(self, "segment_screening_charge_e", charges)
        object.__setattr__(self, "dielectric_energy_hartree", dielectric)
        object.__setattr__(self, "dielectric_energy_role", dielectric_role)
        object.__setattr__(self, "cavity_volume_angstrom3", volume)
        object.__setattr__(self, "molecular_charge_e", molecular_charge)
        object.__setattr__(self, "screening_charge_constraint", charge_constraint)

    @property
    def cavity_area_angstrom2(self):
        return self.segment_areas_angstrom2.sum()


@dataclass(frozen=True, slots=True)
class SigmaProfile:
    """COSMO-RS segment descriptors and molecule-level continuum terms."""

    name: str
    sigma_e_per_angstrom2: Any
    sigma_orthogonal_e_per_angstrom2: Any
    areas_angstrom2: Any
    atomic_numbers: Any
    hydrogen_bond_donor_weight: Any
    hydrogen_bond_acceptor_weight: Any
    cavity_volume_angstrom3: Any
    dielectric_energy_hartree: Any
    dielectric_energy_role: str
    molecular_charge_e: Any
    source_identity: str
    molecule_atomic_numbers: tuple[int, ...]
    discretization: str = "continuous-segments"

    def __post_init__(self) -> None:
        sigma = _float64_tensor(
            self.sigma_e_per_angstrom2,
            name="sigma_e_per_angstrom2",
        )
        device = sigma.device
        orthogonal = _float64_tensor(
            self.sigma_orthogonal_e_per_angstrom2,
            name="sigma_orthogonal_e_per_angstrom2",
            device=device,
        )
        areas = _float64_tensor(
            self.areas_angstrom2,
            name="areas_angstrom2",
            device=device,
        )
        numbers = _int64_tensor(
            self.atomic_numbers,
            name="atomic_numbers",
            device=device,
        )
        donor = _float64_tensor(
            self.hydrogen_bond_donor_weight,
            name="hydrogen_bond_donor_weight",
            device=device,
        )
        acceptor = _float64_tensor(
            self.hydrogen_bond_acceptor_weight,
            name="hydrogen_bond_acceptor_weight",
            device=device,
        )
        volume = _float64_tensor(
            self.cavity_volume_angstrom3,
            name="cavity_volume_angstrom3",
            device=device,
        )
        dielectric = _float64_tensor(
            self.dielectric_energy_hartree,
            name="dielectric_energy_hartree",
            device=device,
        )
        molecular_charge = _float64_tensor(
            self.molecular_charge_e,
            name="molecular_charge_e",
            device=device,
        )
        if sigma.ndim != 1 or sigma.numel() == 0:
            raise ValueError(
                "A sigma profile must contain a nonempty 1-D segment axis."
            )
        if any(
            value.shape != sigma.shape
            for value in (orthogonal, areas, numbers, donor, acceptor)
        ):
            raise ValueError("All sigma-profile descriptors must share shape (S,).")
        if bool((areas <= 0.0).any().detach()):
            raise ValueError("Sigma-profile areas must be positive.")
        if bool(((donor < 0.0) | (acceptor < 0.0)).any().detach()):
            raise ValueError("Hydrogen-bond weights must be nonnegative.")
        if volume.ndim != 0 or dielectric.ndim != 0 or molecular_charge.ndim != 0:
            raise ValueError("Sigma-profile molecule properties must be scalars.")
        if float(volume.detach()) <= 0.0:
            raise ValueError("Sigma-profile cavity volume must be positive.")
        if not str(self.name).strip() or not str(self.source_identity).strip():
            raise ValueError(
                "Sigma-profile name and source identity must be non-empty."
            )
        molecule_numbers = tuple(self.molecule_atomic_numbers)
        if not molecule_numbers or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in molecule_numbers
        ):
            raise ValueError(
                "molecule_atomic_numbers must be a nonempty tuple of positive integers."
            )
        if not str(self.discretization).strip():
            raise ValueError("Sigma-profile discretization must be non-empty.")
        dielectric_role = str(self.dielectric_energy_role).strip()
        if dielectric_role not in {
            "total-solvated-minus-gas",
            "boundary-polarization-only",
        }:
            raise ValueError("Sigma-profile dielectric_energy_role is unsupported.")
        object.__setattr__(self, "sigma_e_per_angstrom2", sigma)
        object.__setattr__(self, "sigma_orthogonal_e_per_angstrom2", orthogonal)
        object.__setattr__(self, "areas_angstrom2", areas)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "hydrogen_bond_donor_weight", donor)
        object.__setattr__(self, "hydrogen_bond_acceptor_weight", acceptor)
        object.__setattr__(self, "cavity_volume_angstrom3", volume)
        object.__setattr__(self, "dielectric_energy_hartree", dielectric)
        object.__setattr__(self, "dielectric_energy_role", dielectric_role)
        object.__setattr__(self, "molecular_charge_e", molecular_charge)
        object.__setattr__(self, "molecule_atomic_numbers", molecule_numbers)

    @property
    def cavity_area_angstrom2(self):
        return self.areas_angstrom2.sum()

    @property
    def segment_counts(self):
        from .cosmospace import OPEN_COSMORS_24A_PARAMETERS

        return self.areas_angstrom2 / (
            OPEN_COSMORS_24A_PARAMETERS.effective_segment_area_angstrom2
        )


def sigma_average(
    segment_positions_angstrom: object,
    segment_areas_angstrom2: object,
    raw_sigma_e_per_angstrom2: object,
    *,
    averaging_radius_angstrom: float,
    squared_distances_angstrom2: object | None = None,
):
    """Apply the published Gaussian COSMO sigma-averaging equation."""

    torch = _torch()
    positions = _float64_tensor(
        segment_positions_angstrom,
        name="segment_positions_angstrom",
    )
    device = positions.device
    areas = _float64_tensor(
        segment_areas_angstrom2,
        name="segment_areas_angstrom2",
        device=device,
    )
    sigma = _float64_tensor(
        raw_sigma_e_per_angstrom2,
        name="raw_sigma_e_per_angstrom2",
        device=device,
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("segment_positions_angstrom must have shape (S, 3).")
    if areas.shape != (positions.shape[0],) or sigma.shape != areas.shape:
        raise ValueError("Sigma averaging requires matching segment arrays.")
    radius = float(averaging_radius_angstrom)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("averaging_radius_angstrom must be finite and positive.")
    if squared_distances_angstrom2 is None:
        distances2 = torch.cdist(positions, positions).square()
    else:
        distances2 = _float64_tensor(
            squared_distances_angstrom2,
            name="squared_distances_angstrom2",
            device=device,
        )
        if distances2.shape != (positions.shape[0], positions.shape[0]):
            raise ValueError("squared_distances_angstrom2 has the wrong shape.")
        if bool((distances2 < 0.0).any().detach()):
            raise ValueError("Squared segment distances cannot be negative.")
    segment_radius2 = areas / math.pi
    inverse_width = 1.0 / (segment_radius2 + radius * radius)
    weights = (
        radius
        * radius
        * inverse_width[:, None]
        * torch.exp(-distances2.T * inverse_width[:, None])
    )
    numerator = (sigma * segment_radius2) @ weights
    denominator = segment_radius2 @ weights
    if bool((denominator <= 0.0).any().detach()):
        raise FloatingPointError("Sigma averaging produced a nonpositive denominator.")
    averaged = numerator / denominator
    if not bool(torch.isfinite(averaged).all().detach()):
        raise FloatingPointError("Sigma averaging produced non-finite values.")
    return averaged, distances2


def build_sigma_profile(surface: COSMOSurface) -> SigmaProfile:
    """Convert a conductor surface to continuous open24a segment descriptors."""

    if not isinstance(surface, COSMOSurface):
        raise TypeError("surface must be COSMOSurface.")
    from .cosmospace import OPEN_COSMORS_24A_PARAMETERS

    raw_sigma = surface.segment_screening_charge_e / surface.segment_areas_angstrom2
    sigma, distances2 = sigma_average(
        surface.segment_positions_angstrom,
        surface.segment_areas_angstrom2,
        raw_sigma,
        averaging_radius_angstrom=(
            OPEN_COSMORS_24A_PARAMETERS.averaging_radius_angstrom
        ),
    )
    corrected, _ = sigma_average(
        surface.segment_positions_angstrom,
        surface.segment_areas_angstrom2,
        raw_sigma,
        averaging_radius_angstrom=OPEN_COSMORS_ORTHOGONAL_RADIUS_ANGSTROM,
        squared_distances_angstrom2=distances2,
    )
    orthogonal = corrected - OPEN_COSMORS_ORTHOGONAL_SUBTRACTION * sigma
    segment_numbers = _torch().as_tensor(
        surface.atomic_numbers,
        dtype=_torch().int64,
        device=sigma.device,
    )[surface.segment_parent_atom_indices]
    # openCOSMO-RS 24a enables both donor and acceptor switches for every
    # parameterized element; the sigma thresholds in the interaction equation
    # select the physical sign continuously.
    hbond_weight = _torch().ones_like(sigma)
    return SigmaProfile(
        name=surface.name,
        sigma_e_per_angstrom2=sigma,
        sigma_orthogonal_e_per_angstrom2=orthogonal,
        areas_angstrom2=surface.segment_areas_angstrom2,
        atomic_numbers=segment_numbers,
        hydrogen_bond_donor_weight=hbond_weight,
        hydrogen_bond_acceptor_weight=hbond_weight,
        cavity_volume_angstrom3=surface.cavity_volume_angstrom3,
        dielectric_energy_hartree=surface.dielectric_energy_hartree,
        dielectric_energy_role=surface.dielectric_energy_role,
        molecular_charge_e=surface.molecular_charge_e,
        source_identity=surface.source_identity,
        molecule_atomic_numbers=surface.atomic_numbers,
    )


def discretize_open24a_profile(profile: SigmaProfile) -> SigmaProfile:
    """Bilinearly place continuous segments on the published 0.001 sigma grid."""

    torch = _torch()
    if not isinstance(profile, SigmaProfile):
        raise TypeError("profile must be SigmaProfile.")
    sigma = profile.sigma_e_per_angstrom2
    orthogonal = profile.sigma_orthogonal_e_per_angstrom2
    lower = OPEN_COSMORS_SIGMA_MIN
    upper = OPEN_COSMORS_SIGMA_MAX
    step = OPEN_COSMORS_SIGMA_STEP
    if bool(
        (
            (sigma < lower)
            | (sigma > upper)
            | (orthogonal < lower)
            | (orthogonal > upper)
        )
        .any()
        .detach()
    ):
        raise ValueError("A sigma descriptor lies outside the open24a grid.")
    interval_count = int(round((upper - lower) / step))

    def brackets(values: Any):
        scaled = (values - lower) / step
        left = torch.floor(scaled).to(dtype=torch.int64)
        left = torch.clamp(left, min=0, max=interval_count - 1)
        right_fraction = torch.clamp(scaled - left, min=0.0, max=1.0)
        return left, left + 1, 1.0 - right_fraction, right_fraction

    sigma_left, sigma_right, sigma_w_left, sigma_w_right = brackets(sigma)
    orth_left, orth_right, orth_w_left, orth_w_right = brackets(orthogonal)
    sigma_indices = torch.stack(
        (sigma_left, sigma_left, sigma_right, sigma_right), dim=1
    ).reshape(-1)
    orth_indices = torch.stack(
        (orth_left, orth_right, orth_left, orth_right), dim=1
    ).reshape(-1)
    interpolation = torch.stack(
        (
            sigma_w_left * orth_w_left,
            sigma_w_left * orth_w_right,
            sigma_w_right * orth_w_left,
            sigma_w_right * orth_w_right,
        ),
        dim=1,
    ).reshape(-1)
    numbers = profile.atomic_numbers[:, None].expand(-1, 4).reshape(-1)
    keys = torch.stack((sigma_indices, orth_indices, numbers), dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    expanded_areas = (
        profile.areas_angstrom2[:, None].expand(-1, 4).reshape(-1) * interpolation
    )
    clustered_areas = torch.zeros(
        unique_keys.shape[0], dtype=torch.float64, device=sigma.device
    ).scatter_add(0, inverse, expanded_areas)
    keep = clustered_areas > 0.0
    unique_keys = unique_keys[keep]
    clustered_areas = clustered_areas[keep]
    clustered_sigma = lower + step * unique_keys[:, 0].to(dtype=torch.float64)
    clustered_orthogonal = lower + step * unique_keys[:, 1].to(dtype=torch.float64)
    weights = torch.ones_like(clustered_sigma)
    result = replace(
        profile,
        sigma_e_per_angstrom2=clustered_sigma,
        sigma_orthogonal_e_per_angstrom2=clustered_orthogonal,
        areas_angstrom2=clustered_areas,
        atomic_numbers=unique_keys[:, 2],
        hydrogen_bond_donor_weight=weights,
        hydrogen_bond_acceptor_weight=weights,
        discretization="open24a-bilinear-0.001",
    )
    if not bool(
        torch.isclose(
            result.cavity_area_angstrom2,
            profile.cavity_area_angstrom2,
            atol=2.0e-12,
            rtol=2.0e-13,
        ).detach()
    ):
        raise RuntimeError("open24a sigma gridding did not conserve surface area.")
    return result


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"


def _number(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def parse_orca_cosmo(
    path: str | Path,
    *,
    gas_energy_hartree: float | None = None,
) -> COSMOSurface:
    """Read an ORCA ``.orcacosmo`` file without invoking an external program."""

    torch = _torch()
    source = Path(path).expanduser().resolve()
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines or ":" not in lines[0]:
        raise ValueError("ORCA COSMO file is missing its identity header.")
    name = lines[0].split(":", 1)[0].strip()

    def one_line_index(fragment: str) -> int:
        matches = [index for index, line in enumerate(lines) if fragment in line]
        if len(matches) != 1:
            raise ValueError(f"ORCA COSMO file must contain one {fragment!r} marker.")
        return matches[0]

    xyz_marker = one_line_index("#XYZ_FILE")
    atom_count = int(lines[xyz_marker + 1].strip())
    atoms = []
    symbols = []
    for line in lines[xyz_marker + 3 : xyz_marker + 3 + atom_count]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("ORCA COSMO XYZ section has an invalid row.")
        symbols.append(fields[0])
        atoms.append([_number(value) for value in fields[1:]])

    radii_marker = one_line_index("CARTESIAN COORDINATES (A.U.)")
    radius_rows = []
    cursor = radii_marker + 1
    while cursor < len(lines) and (not lines[cursor].strip() or "---" in lines[cursor]):
        cursor += 1
    for line in lines[cursor : cursor + atom_count]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("ORCA COSMO atom/radius section has an invalid row.")
        radius_rows.append(fields)
    if all(len(fields) >= 5 for fields in radius_rows):
        atomic_numbers = tuple(int(fields[-1]) for fields in radius_rows)
    else:
        from ase.data import atomic_numbers as ase_atomic_numbers

        try:
            atomic_numbers = tuple(ase_atomic_numbers[symbol] for symbol in symbols)
        except KeyError as exc:
            raise ValueError(
                "ORCA COSMO XYZ section contains an unknown element."
            ) from exc

    count_matches = [
        re.match(r"\s*(\d+)\s+#\s*Number of surface points", line) for line in lines
    ]
    counts = [match for match in count_matches if match is not None]
    if len(counts) != 1:
        raise ValueError("ORCA COSMO file must contain one surface-point count.")
    segment_count = int(counts[0].group(1))
    surface_marker = one_line_index("SURFACE POINTS (A.U.)")
    cursor = surface_marker + 1
    while cursor < len(lines) and not lines[cursor].lstrip().startswith("X"):
        cursor += 1
    cursor += 1
    rows = [lines[index].split() for index in range(cursor, cursor + segment_count)]
    if any(len(row) < 10 for row in rows):
        raise ValueError("ORCA COSMO surface section has an invalid row.")
    points_bohr = [[_number(value) for value in row[:3]] for row in rows]
    areas_bohr2 = [_number(row[3]) for row in rows]
    uncorrected_charges = [_number(row[5]) for row in rows]
    parents = [int(row[-1]) for row in rows]

    corrected_marker = next(
        (
            index
            for index, line in enumerate(lines)
            if "#COSMO_corrected" in line or "#CPCM_corrected" in line
        ),
        None,
    )
    corrected_charges = uncorrected_charges
    charge_constraint = "file-as-provided"
    dielectric_energy = None
    if corrected_marker is not None:
        energy_pattern = re.compile(
            rf"Corrected\s+dielectric\s+energy\s*=\s*({_FLOAT})"
        )
        for line in lines[corrected_marker + 1 :]:
            match = energy_pattern.search(line)
            if match is not None:
                dielectric_energy = _number(match.group(1))
                break
        charge_header = next(
            (
                index
                for index in range(corrected_marker + 1, len(lines))
                if "corrected charges:" in lines[index]
            ),
            None,
        )
        if charge_header is not None:
            corrected_charges = [
                _number(lines[index].strip())
                for index in range(charge_header + 1, charge_header + 1 + segment_count)
            ]
            charge_constraint = "exact-total-charge"
    if dielectric_energy is None:
        pattern = re.compile(rf"\s*({_FLOAT})\s+#\s*CPCM dielectric energy")
        values = [pattern.match(line) for line in lines]
        matches = [match for match in values if match is not None]
        if len(matches) != 1:
            raise ValueError("ORCA COSMO file lacks one dielectric-energy value.")
        dielectric_energy = _number(matches[0].group(1))

    dielectric_role = "boundary-polarization-only"
    if gas_energy_hartree is not None:
        gas_energy = float(gas_energy_hartree)
        if not math.isfinite(gas_energy):
            raise ValueError("gas_energy_hartree must be finite.")
        total_pattern = re.compile(rf"FINAL\s+SINGLE\s+POINT\s+ENERGY\s*({_FLOAT})")
        total_matches = [total_pattern.search(line) for line in lines]
        total_values = [match for match in total_matches if match is not None]
        uncorrected_pattern = re.compile(rf"\s*({_FLOAT})\s+#\s*CPCM dielectric energy")
        uncorrected_matches = [uncorrected_pattern.match(line) for line in lines]
        uncorrected_values = [
            match for match in uncorrected_matches if match is not None
        ]
        if len(total_values) != 1 or len(uncorrected_values) != 1:
            raise ValueError(
                "ORCA COSMO file cannot reconstruct its corrected total energy."
            )
        corrected_total = (
            _number(total_values[0].group(1))
            - _number(uncorrected_values[0].group(1))
            + dielectric_energy
        )
        dielectric_energy = corrected_total - gas_energy
        dielectric_role = "total-solvated-minus-gas"

    def scalar_with_comment(comment: str, power: int) -> float:
        pattern = re.compile(rf"\s*({_FLOAT})\s+#\s*{comment}\s*$")
        matches = [pattern.match(line) for line in lines]
        values = [match for match in matches if match is not None]
        if len(values) != 1:
            raise ValueError(f"ORCA COSMO file lacks one {comment.lower()} value.")
        return _number(values[0].group(1)) * BOHR_ANGSTROM**power

    points = torch.tensor(points_bohr, dtype=torch.float64) * BOHR_ANGSTROM
    areas = torch.tensor(areas_bohr2, dtype=torch.float64) * BOHR_ANGSTROM**2
    charges = torch.tensor(corrected_charges, dtype=torch.float64)
    molecular_charge = torch.round(-charges.sum())
    return COSMOSurface(
        name=name,
        atomic_numbers=atomic_numbers,
        atom_positions_angstrom=torch.tensor(atoms, dtype=torch.float64),
        segment_positions_angstrom=points,
        segment_areas_angstrom2=areas,
        segment_parent_atom_indices=torch.tensor(parents, dtype=torch.int64),
        segment_screening_charge_e=charges,
        dielectric_energy_hartree=torch.tensor(dielectric_energy, dtype=torch.float64),
        dielectric_energy_role=dielectric_role,
        cavity_volume_angstrom3=torch.tensor(
            scalar_with_comment("Volume", 3), dtype=torch.float64
        ),
        molecular_charge_e=molecular_charge,
        source_identity=f"orca-cosmo-file:{source.name}",
        screening_charge_constraint=charge_constraint,
    )


def sigma_profile_payload(profile: SigmaProfile) -> dict[str, object]:
    """Return a portable, non-pickle native profile representation."""

    if not isinstance(profile, SigmaProfile):
        raise TypeError("profile must be SigmaProfile.")

    def vector(value: Any) -> list[float] | list[int]:
        return value.detach().cpu().tolist()

    def scalar(value: Any) -> float:
        return float(value.detach().cpu())

    return {
        "schema_version": 1,
        "format": "maple-torch-cosmors-sigma-profile",
        "name": profile.name,
        "sigma_e_per_angstrom2": vector(profile.sigma_e_per_angstrom2),
        "sigma_orthogonal_e_per_angstrom2": vector(
            profile.sigma_orthogonal_e_per_angstrom2
        ),
        "areas_angstrom2": vector(profile.areas_angstrom2),
        "atomic_numbers": vector(profile.atomic_numbers),
        "hydrogen_bond_donor_weight": vector(profile.hydrogen_bond_donor_weight),
        "hydrogen_bond_acceptor_weight": vector(profile.hydrogen_bond_acceptor_weight),
        "cavity_volume_angstrom3": scalar(profile.cavity_volume_angstrom3),
        "dielectric_energy_hartree": scalar(profile.dielectric_energy_hartree),
        "dielectric_energy_role": profile.dielectric_energy_role,
        "molecular_charge_e": scalar(profile.molecular_charge_e),
        "source_identity": profile.source_identity,
        "molecule_atomic_numbers": list(profile.molecule_atomic_numbers),
        "discretization": profile.discretization,
    }


def write_sigma_profile(profile: SigmaProfile, path: str | Path) -> None:
    """Write a native Torch-COSMO-RS profile as deterministic JSON."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            sigma_profile_payload(profile),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def read_sigma_profile(path: str | Path) -> SigmaProfile:
    """Read a native Torch-COSMO-RS JSON profile without pickle or executables."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("format") != (
        "maple-torch-cosmors-sigma-profile"
    ):
        raise ValueError("Unsupported Torch COSMO-RS sigma-profile schema.")
    torch = _torch()
    return SigmaProfile(
        name=str(payload["name"]),
        sigma_e_per_angstrom2=torch.tensor(
            payload["sigma_e_per_angstrom2"], dtype=torch.float64
        ),
        sigma_orthogonal_e_per_angstrom2=torch.tensor(
            payload["sigma_orthogonal_e_per_angstrom2"], dtype=torch.float64
        ),
        areas_angstrom2=torch.tensor(payload["areas_angstrom2"], dtype=torch.float64),
        atomic_numbers=torch.tensor(payload["atomic_numbers"], dtype=torch.int64),
        hydrogen_bond_donor_weight=torch.tensor(
            payload["hydrogen_bond_donor_weight"], dtype=torch.float64
        ),
        hydrogen_bond_acceptor_weight=torch.tensor(
            payload["hydrogen_bond_acceptor_weight"], dtype=torch.float64
        ),
        cavity_volume_angstrom3=torch.tensor(
            payload["cavity_volume_angstrom3"], dtype=torch.float64
        ),
        dielectric_energy_hartree=torch.tensor(
            payload["dielectric_energy_hartree"], dtype=torch.float64
        ),
        dielectric_energy_role=str(payload["dielectric_energy_role"]),
        molecular_charge_e=torch.tensor(
            payload["molecular_charge_e"], dtype=torch.float64
        ),
        source_identity=str(payload["source_identity"]),
        molecule_atomic_numbers=tuple(payload["molecule_atomic_numbers"]),
        discretization=str(payload["discretization"]),
    )


__all__ = [
    "BOHR_ANGSTROM",
    "COSMOSurface",
    "OPEN_COSMORS_ORTHOGONAL_RADIUS_ANGSTROM",
    "OPEN_COSMORS_ORTHOGONAL_SUBTRACTION",
    "OPEN_COSMORS_SIGMA_MAX",
    "OPEN_COSMORS_SIGMA_MIN",
    "OPEN_COSMORS_SIGMA_STEP",
    "SigmaProfile",
    "build_sigma_profile",
    "discretize_open24a_profile",
    "parse_orca_cosmo",
    "read_sigma_profile",
    "sigma_profile_payload",
    "sigma_average",
    "write_sigma_profile",
]

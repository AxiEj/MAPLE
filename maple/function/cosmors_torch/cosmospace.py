"""Clean-room PyTorch implementation of the public COSMOspace equations.

This module implements only the statistical segment-interaction core.  It
does not parse proprietary COSMOtherm assets and does not claim numerical
identity with a commercial parameterization.  The bundled 24a values are the
published neutral-molecule openCOSMO-RS parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise ImportError("The COSMO-RS Torch backend requires PyTorch.") from exc
    return torch


@dataclass(frozen=True, slots=True)
class COSMOSPACEParameters:
    """Interaction and nonlinear-solver parameters for one model identity."""

    name: str
    effective_segment_area_angstrom2: float
    averaging_radius_angstrom: float
    misfit_alpha_j_angstrom2_per_mol_e2: float
    misfit_orthogonal_factor: float
    hydrogen_bond_coefficient_j_angstrom2_per_mol_e2: float
    hydrogen_bond_temperature_coefficient: float
    hydrogen_bond_sigma_threshold_e_per_angstrom2: float
    reference_temperature_k: float = 298.15
    successive_substitution_mixing: float = 0.7
    convergence_relative: float = 1.0e-6
    maximum_iterations: int = 1000

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("COSMOspace parameterization name must be non-empty.")
        positive = (
            "effective_segment_area_angstrom2",
            "averaging_radius_angstrom",
            "misfit_alpha_j_angstrom2_per_mol_e2",
            "hydrogen_bond_coefficient_j_angstrom2_per_mol_e2",
            "hydrogen_bond_sigma_threshold_e_per_angstrom2",
            "reference_temperature_k",
            "convergence_relative",
        )
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in positive
        ):
            raise ValueError("COSMOspace positive parameters must be finite.")
        if not math.isfinite(self.misfit_orthogonal_factor):
            raise ValueError("COSMOspace orthogonal factor must be finite.")
        if not 0.0 < self.successive_substitution_mixing <= 1.0:
            raise ValueError("COSMOspace mixing must lie in (0, 1].")
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, int)
            or self.maximum_iterations <= 0
        ):
            raise ValueError("COSMOspace maximum_iterations must be positive.")


OPEN_COSMORS_24A_PARAMETERS = COSMOSPACEParameters(
    name="openCOSMO-RS-24a-neutral-ORCA6",
    effective_segment_area_angstrom2=5.9248470,
    averaging_radius_angstrom=0.5,
    misfit_alpha_j_angstrom2_per_mol_e2=7.2847361e6,
    misfit_orthogonal_factor=2.4,
    hydrogen_bond_coefficient_j_angstrom2_per_mol_e2=4.3311555e7,
    hydrogen_bond_temperature_coefficient=1.5,
    hydrogen_bond_sigma_threshold_e_per_angstrom2=9.6112460e-3,
)


@dataclass(frozen=True, slots=True)
class COSMOSPACEResult:
    segment_activity_coefficients: Any
    segment_mole_fractions: Any
    iterations: Any
    maximum_relative_change: Any
    converged: Any


def _as_float64_tensor(value: object, *, name: str):
    torch = _torch()
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} must use torch.float64.")
    if not bool(torch.isfinite(tensor).all().detach()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def build_segment_interaction_energy(
    sigma_e_per_angstrom2: object,
    sigma_orthogonal_e_per_angstrom2: object,
    hydrogen_bond_donor_weight: object,
    hydrogen_bond_acceptor_weight: object,
    *,
    temperature_k: float,
    parameters: COSMOSPACEParameters = OPEN_COSMORS_24A_PARAMETERS,
):
    """Return the symmetric misfit-plus-hydrogen-bond matrix in J/mol."""

    torch = _torch()
    if not isinstance(parameters, COSMOSPACEParameters):
        raise TypeError("parameters must be COSMOSPACEParameters.")
    temperature = float(temperature_k)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_k must be finite and positive.")
    sigma = _as_float64_tensor(sigma_e_per_angstrom2, name="sigma")
    orthogonal = _as_float64_tensor(
        sigma_orthogonal_e_per_angstrom2,
        name="sigma_orthogonal",
    )
    donor = _as_float64_tensor(
        hydrogen_bond_donor_weight,
        name="hydrogen_bond_donor_weight",
    )
    acceptor = _as_float64_tensor(
        hydrogen_bond_acceptor_weight,
        name="hydrogen_bond_acceptor_weight",
    )
    if sigma.ndim != 1 or any(
        value.shape != sigma.shape for value in (orthogonal, donor, acceptor)
    ):
        raise ValueError("COSMO-RS segment descriptors must be equal 1-D arrays.")
    if bool(((donor < 0.0) | (acceptor < 0.0)).any().detach()):
        raise ValueError("Hydrogen-bond weights must be nonnegative.")

    sigma_sum = sigma[:, None] + sigma[None, :]
    orthogonal_sum = orthogonal[:, None] + orthogonal[None, :]
    misfit_prefactor = (
        0.5
        * parameters.misfit_alpha_j_angstrom2_per_mol_e2
        * parameters.effective_segment_area_angstrom2
    )
    misfit = (
        misfit_prefactor
        * sigma_sum
        * (sigma_sum + parameters.misfit_orthogonal_factor * orthogonal_sum)
    )

    threshold = parameters.hydrogen_bond_sigma_threshold_e_per_angstrom2
    donor_strength = donor * torch.clamp(sigma + threshold, max=0.0)
    acceptor_strength = acceptor * torch.clamp(sigma - threshold, min=0.0)
    pair_strength = (
        donor_strength[:, None] * acceptor_strength[None, :]
        + acceptor_strength[:, None] * donor_strength[None, :]
    )
    temperature_factor = (
        1.0
        - parameters.hydrogen_bond_temperature_coefficient
        + parameters.hydrogen_bond_temperature_coefficient
        * parameters.reference_temperature_k
        / temperature
    )
    hb_coefficient = parameters.hydrogen_bond_coefficient_j_angstrom2_per_mol_e2 * max(
        temperature_factor, 0.0
    )
    hydrogen_bond = (
        hb_coefficient * parameters.effective_segment_area_angstrom2 * pair_strength
    )
    result = misfit + hydrogen_bond
    if not bool(torch.isfinite(result).all().detach()):
        raise FloatingPointError("COSMO-RS interaction energy is non-finite.")
    return 0.5 * (result + result.T)


def solve_cosmospace(
    segment_mole_fractions: object,
    interaction_energy_j_per_mol: object,
    *,
    temperature_k: float,
    parameters: COSMOSPACEParameters = OPEN_COSMORS_24A_PARAMETERS,
) -> COSMOSPACEResult:
    """Solve the COSMOspace fixed point, including leading batch dimensions."""

    torch = _torch()
    if not isinstance(parameters, COSMOSPACEParameters):
        raise TypeError("parameters must be COSMOSPACEParameters.")
    temperature = float(temperature_k)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_k must be finite and positive.")
    fractions = _as_float64_tensor(
        segment_mole_fractions,
        name="segment_mole_fractions",
    )
    interaction = _as_float64_tensor(
        interaction_energy_j_per_mol,
        name="interaction_energy_j_per_mol",
    )
    if fractions.ndim < 1 or fractions.shape[-1] == 0:
        raise ValueError("segment_mole_fractions must end in a nonempty axis.")
    segment_count = fractions.shape[-1]
    if interaction.shape[-2:] != (segment_count, segment_count):
        raise ValueError("Interaction matrices must have shape (..., S, S).")
    if interaction.device != fractions.device:
        raise ValueError("COSMOspace inputs must share one Torch device.")
    if bool((fractions < 0.0).any().detach()):
        raise ValueError("Segment mole fractions must be nonnegative.")
    totals = fractions.sum(dim=-1, keepdim=True)
    if bool((totals <= 0.0).any().detach()):
        raise ValueError("Every COSMOspace batch member needs positive area.")
    fractions = fractions / totals
    try:
        batch_shape = torch.broadcast_shapes(
            fractions.shape[:-1],
            interaction.shape[:-2],
        )
    except RuntimeError as exc:
        raise ValueError("COSMOspace batch dimensions are not broadcastable.") from exc
    fractions = fractions.expand(batch_shape + (segment_count,))
    interaction = interaction.expand(batch_shape + (segment_count, segment_count))
    tau = torch.exp(-interaction / (GAS_CONSTANT_J_PER_MOL_K * temperature))
    if not bool(torch.isfinite(tau).all().detach()):
        raise FloatingPointError("COSMOspace Boltzmann matrix is non-finite.")

    gamma = torch.ones_like(fractions)
    active = torch.ones(batch_shape or (), dtype=torch.bool, device=fractions.device)
    iterations = torch.zeros(
        batch_shape or (), dtype=torch.int64, device=fractions.device
    )
    maximum_relative_change = torch.full_like(
        iterations,
        float("inf"),
        dtype=torch.float64,
    )
    for iteration in range(1, parameters.maximum_iterations + 1):
        denominator = torch.matmul(
            tau,
            (fractions * gamma).unsqueeze(-1),
        ).squeeze(-1)
        if bool((denominator <= 0.0).any().detach()):
            raise FloatingPointError("COSMOspace denominator became nonpositive.")
        proposal = denominator.reciprocal()
        relative = torch.amax(torch.abs(proposal - gamma) / gamma, dim=-1)
        newly_converged = active & (relative < parameters.convergence_relative)
        iterations = torch.where(
            newly_converged,
            torch.full_like(iterations, iteration),
            iterations,
        )
        maximum_relative_change = torch.where(
            active,
            relative,
            maximum_relative_change,
        )
        active = active & ~newly_converged
        if not bool(active.any().detach()):
            return COSMOSPACEResult(
                segment_activity_coefficients=gamma,
                segment_mole_fractions=fractions,
                iterations=iterations,
                maximum_relative_change=maximum_relative_change,
                converged=~active,
            )
        mixed = gamma + parameters.successive_substitution_mixing * (proposal - gamma)
        gamma = torch.where(active.unsqueeze(-1), mixed, gamma)
    raise RuntimeError("COSMOspace did not converge within the iteration cap.")


def molecule_residual_log_activity(
    segment_activity_coefficients: object,
    molecule_segment_counts: object,
):
    """Return residual molecular ``ln(gamma)`` from segment counts."""

    torch = _torch()
    gamma = _as_float64_tensor(
        segment_activity_coefficients,
        name="segment_activity_coefficients",
    )
    counts = _as_float64_tensor(
        molecule_segment_counts,
        name="molecule_segment_counts",
    )
    if gamma.ndim < 1 or counts.shape[-1] != gamma.shape[-1]:
        raise ValueError("Segment counts and activity coefficients must share S.")
    if bool((gamma <= 0.0).any().detach()) or bool((counts < 0.0).any().detach()):
        raise ValueError(
            "Activity coefficients must be positive and counts nonnegative."
        )
    result = torch.matmul(counts, torch.log(gamma).unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(result).all().detach()):
        raise FloatingPointError("Molecular residual activity is non-finite.")
    return result


__all__ = [
    "GAS_CONSTANT_J_PER_MOL_K",
    "OPEN_COSMORS_24A_PARAMETERS",
    "COSMOSPACEParameters",
    "COSMOSPACEResult",
    "build_segment_interaction_energy",
    "molecule_residual_log_activity",
    "solve_cosmospace",
]

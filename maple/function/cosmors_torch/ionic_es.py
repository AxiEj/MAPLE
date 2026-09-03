"""Published COSMO-RS-ES short-range terms for ionic solutes.

This module implements only the polyatomic-anion/neutral-solvent contact
equations from Mueller's COSMO-RS-ES Parameterization C (Table 6.12,
equations 6.25 and 6.26; Table 6.13).  It deliberately excludes the
Pitzer-Debye-Hueckel term, which vanishes at infinite dilution, and the
single-ion reference-scale corrections, which are adjustments to target data
rather than forward-model terms.

The published numerical parameters were fitted with a different COSMO surface
convention and neutral COSMO-RS parameterization.  Applying them to a MACE-EF
surface is therefore an explicit, non-admitted transfer experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

from .cosmospace import COSMOSPACEParameters

IONIC_ES_PAPER_DOI = "10.1021/acs.jpca.0c01606"
IONIC_ES_THESIS_URL = (
    "https://tore.tuhh.de/bitstream/11420/7623/3/"
    "Dissertation.Simon.Mueller.Application.and.Refinement.of.COSMO-RS-ES.pdf"
)
POLYATOMIC_ANION_SHORT_RANGE_IDENTITY = (
    "cosmo-rs-es-parameterization-c-polyatomic-anion-sr-2020"
)
PARAMETERIZATION_C_NEUTRAL_IDENTITY = "cosmo-rs-es-parameterization-c-neutral-2020"
MACE_EF_ELECTRON_ATTACHMENT_LOCALIZATION_IDENTITY = (
    "mace-ef-electron-attachment-fraction-localization-v1"
)
ELECTRON_ATTACHMENT_SMOOTHING_E = 1.0e-8
IonicESSolventClass = Literal["water", "organic"]

PARAMETERIZATION_C_COMBINATORIAL_STANDARD_AREA_ANGSTROM2 = 116.85
PARAMETERIZATION_C_COMBINATORIAL_VOLUME_EXPONENT = 0.645
PARAMETERIZATION_C_NEUTRAL_HB_DONOR_ATOMIC_NUMBERS = (1,)
PARAMETERIZATION_C_NEUTRAL_HB_ACCEPTOR_ATOMIC_NUMBERS = (
    6,
    7,
    8,
    9,
    15,
    16,
    17,
    35,
    53,
)

PUBLISHED_PARAMETERIZATION_C_NEUTRAL_COSMOSPACE = COSMOSPACEParameters(
    name=PARAMETERIZATION_C_NEUTRAL_IDENTITY,
    effective_segment_area_angstrom2=6.25,
    averaging_radius_angstrom=0.5,
    misfit_alpha_j_angstrom2_per_mol_e2=4.58386e6,
    misfit_orthogonal_factor=2.4,
    hydrogen_bond_coefficient_j_angstrom2_per_mol_e2=1.961622e7,
    hydrogen_bond_temperature_coefficient=1.5,
    hydrogen_bond_sigma_threshold_e_per_angstrom2=0.0085,
)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise ImportError("The ionic COSMO-RS-ES terms require PyTorch.") from exc
    return torch


def _float64_vector(value: object, *, name: str):
    torch = _torch()
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    if tensor.dtype != torch.float64 or tensor.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional torch.float64 tensor.")
    if not bool(torch.isfinite(tensor).all().detach()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


@dataclass(frozen=True, slots=True)
class PolyatomicAnionShortRangeParameters:
    """Parameterization C values in units used by the Torch COSMOspace core."""

    effective_segment_area_angstrom2: float = 6.25
    orthogonal_misfit_factor: float = 2.4
    hydrogen_bond_threshold_e_per_angstrom2: float = 0.0085
    water_misfit_j_angstrom2_per_mol_e2: float = 473.0e3
    organic_misfit_j_angstrom2_per_mol_e2: float = 5408.0e3
    water_attraction_j_angstrom2_per_mol_e2: float = 25454.0e3
    organic_attraction_j_angstrom2_per_mol_e2: float = 143278.0e3
    anion_threshold_e_per_angstrom2: float = 0.0175
    organic_threshold_e_per_angstrom2: float = 0.0094

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": POLYATOMIC_ANION_SHORT_RANGE_IDENTITY,
            "paper_doi": IONIC_ES_PAPER_DOI,
            "thesis_url": IONIC_ES_THESIS_URL,
            "source_tables": ["6.12", "6.13"],
            "source_equations": ["6.25", "6.26"],
            "pdh_long_range_included": False,
            "single_ion_reference_scale_correction_included": False,
            **asdict(self),
        }


PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE = PolyatomicAnionShortRangeParameters()


def electron_attachment_fractions(
    anion_atom_monopoles_e: object,
    neutral_reference_atom_monopoles_e: object,
    *,
    smoothing_e: float = ELECTRON_ATTACHMENT_SMOOTHING_E,
):
    """Return a smooth, normalized per-atom distribution of one added electron."""

    torch = _torch()
    anion = _float64_vector(
        anion_atom_monopoles_e,
        name="anion_atom_monopoles_e",
    )
    reference = _float64_vector(
        neutral_reference_atom_monopoles_e,
        name="neutral_reference_atom_monopoles_e",
    )
    if reference.shape != anion.shape or reference.device != anion.device:
        raise ValueError("Anion and reference monopoles must share shape and device.")
    smoothing = float(smoothing_e)
    if not math.isfinite(smoothing) or smoothing <= 0.0:
        raise ValueError("smoothing_e must be finite and positive.")
    delta = anion - reference
    added_electron = -float(delta.sum().detach())
    if not math.isclose(added_electron, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise ValueError(
            "Electron-attachment localization requires a one-electron charge change."
        )
    smooth_negative = 0.5 * (torch.sqrt(delta.square() + smoothing * smoothing) - delta)
    total = smooth_negative.sum()
    if not bool(torch.isfinite(total).detach()) or float(total.detach()) <= 0.0:
        raise FloatingPointError("Electron-attachment weights cannot be normalized.")
    weights = smooth_negative / total
    if not bool(torch.isfinite(weights).all().detach()):
        raise FloatingPointError("Electron-attachment weights are non-finite.")
    return weights


def parameterization_c_neutral_provenance() -> dict[str, object]:
    """Return the frozen published neutral-baseline identity and values."""

    parameters = PUBLISHED_PARAMETERIZATION_C_NEUTRAL_COSMOSPACE
    return {
        "identity": PARAMETERIZATION_C_NEUTRAL_IDENTITY,
        "paper_doi": IONIC_ES_PAPER_DOI,
        "thesis_url": IONIC_ES_THESIS_URL,
        "source_table": "6.11",
        "source_equations": ["6.21", "2.39", "2.41"],
        "combinatorial_standard_area_angstrom2": (
            PARAMETERIZATION_C_COMBINATORIAL_STANDARD_AREA_ANGSTROM2
        ),
        "combinatorial_volume_exponent": (
            PARAMETERIZATION_C_COMBINATORIAL_VOLUME_EXPONENT
        ),
        "neutral_hbond_donor_atomic_numbers": list(
            PARAMETERIZATION_C_NEUTRAL_HB_DONOR_ATOMIC_NUMBERS
        ),
        "neutral_hbond_acceptor_atomic_numbers": list(
            PARAMETERIZATION_C_NEUTRAL_HB_ACCEPTOR_ATOMIC_NUMBERS
        ),
        "published_cosmospace_parameters": {
            "effective_segment_area_angstrom2": (
                parameters.effective_segment_area_angstrom2
            ),
            "averaging_radius_angstrom": parameters.averaging_radius_angstrom,
            "misfit_alpha_j_angstrom2_per_mol_e2": (
                parameters.misfit_alpha_j_angstrom2_per_mol_e2
            ),
            "misfit_orthogonal_factor": parameters.misfit_orthogonal_factor,
            "hydrogen_bond_coefficient_j_angstrom2_per_mol_e2": (
                parameters.hydrogen_bond_coefficient_j_angstrom2_per_mol_e2
            ),
            "hydrogen_bond_temperature_coefficient": (
                parameters.hydrogen_bond_temperature_coefficient
            ),
            "hydrogen_bond_sigma_threshold_e_per_angstrom2": (
                parameters.hydrogen_bond_sigma_threshold_e_per_angstrom2
            ),
        },
        "numerical_solver": {
            "successive_substitution_mixing": (
                parameters.successive_substitution_mixing
            ),
            "convergence_relative": parameters.convergence_relative,
            "maximum_iterations": parameters.maximum_iterations,
        },
        "open24a_solute_only_ledger_retained": True,
    }


def parameterization_c_neutral_hbond_weights(atomic_numbers: object):
    """Return Parameterization C donor/acceptor switches by parent element."""

    torch = _torch()
    numbers = (
        atomic_numbers
        if isinstance(atomic_numbers, torch.Tensor)
        else torch.as_tensor(atomic_numbers)
    )
    if numbers.dtype != torch.int64 or numbers.ndim != 1:
        raise TypeError("atomic_numbers must be a one-dimensional torch.int64 tensor.")
    donor = numbers == PARAMETERIZATION_C_NEUTRAL_HB_DONOR_ATOMIC_NUMBERS[0]
    acceptor = torch.zeros_like(donor)
    for atomic_number in PARAMETERIZATION_C_NEUTRAL_HB_ACCEPTOR_ATOMIC_NUMBERS:
        acceptor |= numbers == atomic_number
    return donor.to(dtype=torch.float64), acceptor.to(dtype=torch.float64)


def polyatomic_anion_neutral_solvent_cross_energy(
    anion_sigma_e_per_angstrom2: object,
    anion_sigma_orthogonal_e_per_angstrom2: object,
    solvent_sigma_e_per_angstrom2: object,
    solvent_sigma_orthogonal_e_per_angstrom2: object,
    *,
    solvent_class: IonicESSolventClass,
    parameters: PolyatomicAnionShortRangeParameters = (
        PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE
    ),
):
    """Return the Parameterization C anion-solvent cross-contact energy.

    The returned matrix has shape ``(anion segments, solvent segments)`` and
    contains the contact-specific misfit term plus the published attractive
    ionic term.  It replaces, rather than augments, the neutral COSMO-RS
    interaction for those cross contacts.
    """

    torch = _torch()
    if solvent_class not in ("water", "organic"):
        raise ValueError("solvent_class must be 'water' or 'organic'.")
    if not isinstance(parameters, PolyatomicAnionShortRangeParameters):
        raise TypeError("parameters must be PolyatomicAnionShortRangeParameters.")
    anion_sigma = _float64_vector(
        anion_sigma_e_per_angstrom2,
        name="anion_sigma_e_per_angstrom2",
    )
    anion_orthogonal = _float64_vector(
        anion_sigma_orthogonal_e_per_angstrom2,
        name="anion_sigma_orthogonal_e_per_angstrom2",
    )
    solvent_sigma = _float64_vector(
        solvent_sigma_e_per_angstrom2,
        name="solvent_sigma_e_per_angstrom2",
    )
    solvent_orthogonal = _float64_vector(
        solvent_sigma_orthogonal_e_per_angstrom2,
        name="solvent_sigma_orthogonal_e_per_angstrom2",
    )
    if anion_orthogonal.shape != anion_sigma.shape:
        raise ValueError("Anion sigma descriptors must have matching shapes.")
    if solvent_orthogonal.shape != solvent_sigma.shape:
        raise ValueError("Solvent sigma descriptors must have matching shapes.")
    if anion_sigma.device != solvent_sigma.device or any(
        tensor.device != anion_sigma.device
        for tensor in (anion_orthogonal, solvent_orthogonal)
    ):
        raise ValueError("Ionic short-range descriptors must share one device.")

    sigma_sum = anion_sigma[:, None] + solvent_sigma[None, :]
    orthogonal_sum = anion_orthogonal[:, None] + solvent_orthogonal[None, :]
    if solvent_class == "water":
        misfit_coefficient = parameters.water_misfit_j_angstrom2_per_mol_e2
        attraction_coefficient = parameters.water_attraction_j_angstrom2_per_mol_e2
        solvent_strength = torch.clamp(
            solvent_sigma + parameters.hydrogen_bond_threshold_e_per_angstrom2,
            max=0.0,
        )
        anion_strength = torch.clamp(
            anion_sigma - parameters.hydrogen_bond_threshold_e_per_angstrom2,
            min=0.0,
        )
    else:
        misfit_coefficient = parameters.organic_misfit_j_angstrom2_per_mol_e2
        attraction_coefficient = parameters.organic_attraction_j_angstrom2_per_mol_e2
        solvent_strength = torch.clamp(
            solvent_sigma + parameters.organic_threshold_e_per_angstrom2,
            max=0.0,
        )
        anion_strength = torch.clamp(
            anion_sigma - parameters.anion_threshold_e_per_angstrom2,
            min=0.0,
        )

    half_area = 0.5 * parameters.effective_segment_area_angstrom2
    misfit = (
        half_area
        * misfit_coefficient
        * sigma_sum
        * (sigma_sum + parameters.orthogonal_misfit_factor * orthogonal_sum)
    )
    attraction = (
        half_area
        * attraction_coefficient
        * anion_strength[:, None]
        * solvent_strength[None, :]
    )
    result = misfit + attraction
    if not bool(torch.isfinite(result).all().detach()):
        raise FloatingPointError("Ionic COSMO-RS-ES cross energy is non-finite.")
    return result


def replace_polyatomic_anion_cross_contacts(
    neutral_interaction_energy_j_per_mol: object,
    *,
    solute_segment_count: int,
    solute_sigma_e_per_angstrom2: object,
    solute_sigma_orthogonal_e_per_angstrom2: object,
    solvent_sigma_e_per_angstrom2: object,
    solvent_sigma_orthogonal_e_per_angstrom2: object,
    solvent_class: IonicESSolventClass,
    solute_ionic_contact_weight: object | None = None,
    parameters: PolyatomicAnionShortRangeParameters = (
        PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE
    ),
):
    """Replace solute-solvent blocks with the published ionic contact model."""

    torch = _torch()
    base = (
        neutral_interaction_energy_j_per_mol
        if isinstance(neutral_interaction_energy_j_per_mol, torch.Tensor)
        else torch.as_tensor(neutral_interaction_energy_j_per_mol)
    )
    if base.dtype != torch.float64 or base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise TypeError("neutral_interaction_energy_j_per_mol must be square float64.")
    if not 0 < solute_segment_count < base.shape[0]:
        raise ValueError("solute_segment_count does not split the interaction matrix.")
    ionic_cross = polyatomic_anion_neutral_solvent_cross_energy(
        solute_sigma_e_per_angstrom2,
        solute_sigma_orthogonal_e_per_angstrom2,
        solvent_sigma_e_per_angstrom2,
        solvent_sigma_orthogonal_e_per_angstrom2,
        solvent_class=solvent_class,
        parameters=parameters,
    )
    expected = (solute_segment_count, base.shape[0] - solute_segment_count)
    if ionic_cross.shape != expected or ionic_cross.device != base.device:
        raise ValueError("Ionic cross block is incompatible with the base matrix.")
    if solute_ionic_contact_weight is None:
        weights = base.new_ones(solute_segment_count)
    else:
        weights = _float64_vector(
            solute_ionic_contact_weight,
            name="solute_ionic_contact_weight",
        )
        if weights.shape != (solute_segment_count,) or weights.device != base.device:
            raise ValueError(
                "Ionic contact weights must match the solute segment axis."
            )
        if bool(((weights < 0.0) | (weights > 1.0)).any().detach()):
            raise ValueError("Ionic contact weights must lie in [0, 1].")
    neutral_cross = base[:solute_segment_count, solute_segment_count:]
    cross = neutral_cross + weights[:, None] * (ionic_cross - neutral_cross)
    upper = torch.cat(
        (base[:solute_segment_count, :solute_segment_count], cross), dim=1
    )
    lower = torch.cat(
        (cross.T, base[solute_segment_count:, solute_segment_count:]), dim=1
    )
    result = torch.cat((upper, lower), dim=0)
    return 0.5 * (result + result.T)


__all__ = [
    "IONIC_ES_PAPER_DOI",
    "IONIC_ES_THESIS_URL",
    "IonicESSolventClass",
    "ELECTRON_ATTACHMENT_SMOOTHING_E",
    "MACE_EF_ELECTRON_ATTACHMENT_LOCALIZATION_IDENTITY",
    "PARAMETERIZATION_C_COMBINATORIAL_STANDARD_AREA_ANGSTROM2",
    "PARAMETERIZATION_C_COMBINATORIAL_VOLUME_EXPONENT",
    "PARAMETERIZATION_C_NEUTRAL_HB_ACCEPTOR_ATOMIC_NUMBERS",
    "PARAMETERIZATION_C_NEUTRAL_HB_DONOR_ATOMIC_NUMBERS",
    "PARAMETERIZATION_C_NEUTRAL_IDENTITY",
    "POLYATOMIC_ANION_SHORT_RANGE_IDENTITY",
    "PUBLISHED_PARAMETERIZATION_C_NEUTRAL_COSMOSPACE",
    "PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE",
    "PolyatomicAnionShortRangeParameters",
    "electron_attachment_fractions",
    "parameterization_c_neutral_hbond_weights",
    "parameterization_c_neutral_provenance",
    "polyatomic_anion_neutral_solvent_cross_energy",
    "replace_polyatomic_anion_cross_contacts",
]

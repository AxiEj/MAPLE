"""Complete neutral openCOSMO-RS 24a solvation thermodynamics in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

from .cosmospace import (
    GAS_CONSTANT_J_PER_MOL_K,
    OPEN_COSMORS_24A_PARAMETERS,
    COSMOSPACEResult,
    build_segment_interaction_energy,
    molecule_residual_log_activity,
    solve_cosmospace,
)
from .ionic_es import (
    IonicESSolventClass,
    POLYATOMIC_ANION_SHORT_RANGE_IDENTITY,
    PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE,
    replace_polyatomic_anion_cross_contacts,
)
from .surface import SigmaProfile, discretize_open24a_profile

HARTREE_TO_KCAL_PER_MOL = 627.5094740631
JOULE_TO_KCAL = 1.0 / 4184.0


@dataclass(frozen=True, slots=True)
class OpenCOSMORS24aSolvationParameters:
    """Published 24a parameters needed beyond the COSMOspace interaction."""

    combinatorial_standard_area_angstrom2: float = 41.623570
    combinatorial_coordination_number: float = 10.0
    eta_kcal_mol: float = -4.448499
    ring_atom_kcal_mol: float = 0.26302510
    reference_pressure_pa: float = 101325.0
    element_surface_tension_kcal_mol_angstrom2: Mapping[int, float] = field(
        default_factory=lambda: MappingProxyType(
            {
                1: 2.933803e-2,
                6: 2.287904e-2,
                7: 7.007681e-4,
                8: 3.545052e-3,
                9: 5.608829e-3,
                14: 4.215503e-3,
                15: 3.607977e-3,
                16: 3.498700e-2,
                17: 3.414282e-2,
                35: 4.085111e-2,
                53: 2.13e-1,
            }
        )
    )

    def __post_init__(self) -> None:
        positive = (
            self.combinatorial_standard_area_angstrom2,
            self.combinatorial_coordination_number,
            self.ring_atom_kcal_mol,
            self.reference_pressure_pa,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Open24a positive solvation parameters must be finite.")
        if not math.isfinite(self.eta_kcal_mol):
            raise ValueError("Open24a eta must be finite.")
        tensions = dict(self.element_surface_tension_kcal_mol_angstrom2)
        if not tensions or any(
            isinstance(number, bool)
            or not isinstance(number, int)
            or number <= 0
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for number, value in tensions.items()
        ):
            raise ValueError("Open24a element surface tensions are invalid.")
        object.__setattr__(
            self,
            "element_surface_tension_kcal_mol_angstrom2",
            MappingProxyType(tensions),
        )


OPEN_COSMORS_24A_SOLVATION_PARAMETERS = OpenCOSMORS24aSolvationParameters()


@dataclass(frozen=True, slots=True)
class InfiniteDilutionActivity:
    residual_log_activity: Any
    combinatorial_log_activity: Any
    total_log_activity: Any
    cosmospace: COSMOSPACEResult
    solute_profile: SigmaProfile
    solvent_profile: SigmaProfile
    interaction_model_identity: str


@dataclass(frozen=True, slots=True)
class OpenCOSMORS24aSolvationResult:
    delta_g_solvation_kcal_mol: Any
    dielectric_kcal_mol: Any
    chemical_potential_kcal_mol: Any
    element_surface_kcal_mol: Any
    ring_kcal_mol: Any
    reference_state_kcal_mol: Any
    eta_kcal_mol: Any
    solvent_liquid_molar_volume_cm3_mol: Any
    activity: InfiniteDilutionActivity
    ionic_parameterization_validated: bool
    parameterization_identity: str = "openCOSMO-RS-24a-neutral-ORCA6"

    @property
    def diagnostic_only(self) -> bool:
        return not self.ionic_parameterization_validated

    def as_dict(self) -> dict[str, object]:
        def scalar(value: Any) -> float:
            return (
                float(value.detach().cpu())
                if hasattr(value, "detach")
                else float(value)
            )

        return {
            "schema_version": 1,
            "parameterization_identity": self.parameterization_identity,
            "delta_g_solvation_kcal_mol": scalar(self.delta_g_solvation_kcal_mol),
            "ledger_kcal_mol": {
                "dielectric": scalar(self.dielectric_kcal_mol),
                "chemical_potential": scalar(self.chemical_potential_kcal_mol),
                "minus_element_surface": -scalar(self.element_surface_kcal_mol),
                "minus_ring": -scalar(self.ring_kcal_mol),
                "minus_reference_state": -scalar(self.reference_state_kcal_mol),
                "minus_eta": -scalar(self.eta_kcal_mol),
            },
            "log_activity": {
                "residual": scalar(self.activity.residual_log_activity),
                "combinatorial": scalar(self.activity.combinatorial_log_activity),
                "total": scalar(self.activity.total_log_activity),
            },
            "interaction_model_identity": (self.activity.interaction_model_identity),
            "solvent_liquid_molar_volume_cm3_mol": scalar(
                self.solvent_liquid_molar_volume_cm3_mol
            ),
            "ionic_parameterization_validated": self.ionic_parameterization_validated,
            "diagnostic_only": self.diagnostic_only,
            "cosmospace_iterations": int(
                self.activity.cosmospace.iterations.detach().cpu()
            ),
            "profile_discretization": self.activity.solute_profile.discretization,
        }


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise ImportError("Torch COSMO-RS thermodynamics requires PyTorch.") from exc
    return torch


def _same_device(solute: SigmaProfile, solvent: SigmaProfile) -> None:
    if solute.sigma_e_per_angstrom2.device != solvent.sigma_e_per_angstrom2.device:
        raise ValueError("Solute and solvent sigma profiles must use one Torch device.")


def staverman_guggenheim_infinite_dilution(
    solute_volume_angstrom3: object,
    solvent_volume_angstrom3: object,
    solute_area_angstrom2: object,
    solvent_area_angstrom2: object,
    *,
    parameters: OpenCOSMORS24aSolvationParameters = (
        OPEN_COSMORS_24A_SOLVATION_PARAMETERS
    ),
):
    """Return the solute combinatorial ``ln(gamma)`` at infinite dilution."""

    torch = _torch()
    reference = (
        solute_volume_angstrom3
        if isinstance(solute_volume_angstrom3, torch.Tensor)
        else torch.as_tensor(solute_volume_angstrom3, dtype=torch.float64)
    )
    values = []
    for value in (
        solute_volume_angstrom3,
        solvent_volume_angstrom3,
        solute_area_angstrom2,
        solvent_area_angstrom2,
    ):
        tensor = (
            value if isinstance(value, torch.Tensor) else reference.new_tensor(value)
        )
        tensor = tensor.to(dtype=torch.float64, device=reference.device)
        if (
            tensor.ndim != 0
            or not bool(torch.isfinite(tensor).detach())
            or float(tensor.detach()) <= 0.0
        ):
            raise ValueError("Staverman-Guggenheim inputs must be positive scalars.")
        values.append(tensor)
    solute_volume, solvent_volume, solute_area, solvent_area = values
    phi_prime = solute_volume / solvent_volume
    theta_prime = solute_area / solvent_area
    ratio = phi_prime / theta_prime
    relative_area = solute_area / parameters.combinatorial_standard_area_angstrom2
    result = (
        torch.log(phi_prime)
        + 1.0
        - phi_prime
        - 0.5
        * parameters.combinatorial_coordination_number
        * relative_area
        * (torch.log(ratio) + 1.0 - ratio)
    )
    if not bool(torch.isfinite(result).detach()):
        raise FloatingPointError("Staverman-Guggenheim result is non-finite.")
    return result


def infinite_dilution_activity(
    solute_profile: SigmaProfile,
    solvent_profile: SigmaProfile,
    *,
    temperature_k: float = 298.15,
    discretize: bool = True,
    ionic_es_solvent_class: IonicESSolventClass | None = None,
) -> InfiniteDilutionActivity:
    """Solve residual plus combinatorial activity of a solute in pure solvent."""

    torch = _torch()
    if not isinstance(solute_profile, SigmaProfile) or not isinstance(
        solvent_profile, SigmaProfile
    ):
        raise TypeError("solute_profile and solvent_profile must be SigmaProfile.")
    _same_device(solute_profile, solvent_profile)
    temperature = float(temperature_k)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_k must be finite and positive.")
    if abs(float(solvent_profile.molecular_charge_e.detach())) > 2.0e-8:
        raise ValueError("The open24a solvent profile must be neutral.")
    if ionic_es_solvent_class is not None:
        solute_charge = float(solute_profile.molecular_charge_e.detach())
        if solute_charge >= -2.0e-8:
            raise ValueError(
                "The Parameterization C short-range overlay requires an anion."
            )
        if len(solute_profile.molecule_atomic_numbers) < 2:
            raise ValueError(
                "Only the published polyatomic-anion contact class is supported."
            )
        solvent_is_water = sorted(solvent_profile.molecule_atomic_numbers) == [1, 1, 8]
        if (ionic_es_solvent_class == "water") != solvent_is_water:
            raise ValueError(
                "ionic_es_solvent_class disagrees with the solvent profile identity."
            )
    solute = (
        discretize_open24a_profile(solute_profile) if discretize else solute_profile
    )
    solvent = (
        discretize_open24a_profile(solvent_profile) if discretize else solvent_profile
    )
    descriptors = tuple(
        torch.cat((getattr(solute, name), getattr(solvent, name)))
        for name in (
            "sigma_e_per_angstrom2",
            "sigma_orthogonal_e_per_angstrom2",
            "hydrogen_bond_donor_weight",
            "hydrogen_bond_acceptor_weight",
        )
    )
    interaction = build_segment_interaction_energy(
        *descriptors,
        temperature_k=temperature,
    )
    solute_count = solute.areas_angstrom2.numel()
    interaction_identity = OPEN_COSMORS_24A_PARAMETERS.name
    if ionic_es_solvent_class is not None:
        interaction = replace_polyatomic_anion_cross_contacts(
            interaction,
            solute_segment_count=solute_count,
            solute_sigma_e_per_angstrom2=solute.sigma_e_per_angstrom2,
            solute_sigma_orthogonal_e_per_angstrom2=(
                solute.sigma_orthogonal_e_per_angstrom2
            ),
            solvent_sigma_e_per_angstrom2=solvent.sigma_e_per_angstrom2,
            solvent_sigma_orthogonal_e_per_angstrom2=(
                solvent.sigma_orthogonal_e_per_angstrom2
            ),
            solvent_class=ionic_es_solvent_class,
        )
        interaction_identity = (
            f"hybrid-{OPEN_COSMORS_24A_PARAMETERS.name}"
            f"+{POLYATOMIC_ANION_SHORT_RANGE_IDENTITY}"
        )
    solvent_areas = torch.cat(
        (
            torch.zeros(
                solute_count,
                dtype=torch.float64,
                device=solute.areas_angstrom2.device,
            ),
            solvent.areas_angstrom2,
        )
    )
    cosmospace = solve_cosmospace(
        solvent_areas,
        interaction,
        temperature_k=temperature,
    )
    segment_area = (
        PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE.effective_segment_area_angstrom2
        if ionic_es_solvent_class is not None
        else OPEN_COSMORS_24A_PARAMETERS.effective_segment_area_angstrom2
    )
    counts = torch.cat(
        (
            solute.areas_angstrom2 / segment_area,
            torch.zeros_like(solvent.areas_angstrom2),
        )
    )
    residual = molecule_residual_log_activity(
        cosmospace.segment_activity_coefficients,
        counts,
    )
    combinatorial = staverman_guggenheim_infinite_dilution(
        solute.cavity_volume_angstrom3,
        solvent.cavity_volume_angstrom3,
        solute.cavity_area_angstrom2,
        solvent.cavity_area_angstrom2,
    )
    return InfiniteDilutionActivity(
        residual_log_activity=residual,
        combinatorial_log_activity=combinatorial,
        total_log_activity=residual + combinatorial,
        cosmospace=cosmospace,
        solute_profile=solute,
        solvent_profile=solvent,
        interaction_model_identity=interaction_identity,
    )


def estimate_open24a_liquid_molar_volume_cm3_mol(profile: SigmaProfile):
    """Evaluate the published 25 C open24a liquid-volume QSPR."""

    torch = _torch()
    if not isinstance(profile, SigmaProfile):
        raise TypeError("profile must be SigmaProfile.")
    molecule_numbers = profile.molecule_atomic_numbers
    is_water = sorted(molecule_numbers) == [1, 1, 8]
    if is_water:
        return profile.cavity_volume_angstrom3.new_tensor(18.06863632)
    sigma = profile.sigma_e_per_angstrom2
    area = profile.areas_angstrom2
    second_moment = torch.sum(sigma.square() * area) * 100.0**2
    fourth_moment = torch.sum(sigma**4 * area) * 100.0**4
    atom_count = len(molecule_numbers)
    silicon_count = sum(value == 14 for value in molecule_numbers)
    result = (
        0.9430785419976806 * atom_count
        + 0.6977322963011842 * profile.cavity_area_angstrom2
        - 0.3161763939689293 * second_moment
        + 0.032441059832647084 * fourth_moment
        + 8.113026329415828 * silicon_count
        - 0.07066832029215675
    )
    if not bool(torch.isfinite(result).detach()) or float(result.detach()) <= 0.0:
        raise FloatingPointError(
            "Open24a liquid-volume QSPR returned an invalid value."
        )
    return result


def _element_surface_energy(
    profile: SigmaProfile,
    parameters: OpenCOSMORS24aSolvationParameters,
):
    torch = _torch()
    tensions = parameters.element_surface_tension_kcal_mol_angstrom2
    observed = set(
        int(value) for value in torch.unique(profile.atomic_numbers).detach().cpu()
    )
    missing = sorted(observed - set(tensions))
    if missing:
        raise ValueError(
            "No open24a solvation surface parameter is available for atomic numbers: "
            + ", ".join(str(value) for value in missing)
        )
    coefficient = torch.zeros_like(profile.areas_angstrom2)
    for atomic_number, value in tensions.items():
        coefficient = torch.where(
            profile.atomic_numbers == atomic_number,
            coefficient.new_tensor(abs(value)),
            coefficient,
        )
    return torch.sum(coefficient * profile.areas_angstrom2)


def open24a_solvation_free_energy(
    solute_profile: SigmaProfile,
    solvent_profile: SigmaProfile,
    *,
    temperature_k: float = 298.15,
    ring_atom_count: int = 0,
    solvent_liquid_molar_volume_cm3_mol: float | object | None = None,
    total_solvated_minus_gas_hartree: float | object | None = None,
    acknowledge_unvalidated_ions: bool = False,
    discretize: bool = True,
    ionic_es_solvent_class: IonicESSolventClass | None = None,
    parameters: OpenCOSMORS24aSolvationParameters = (
        OPEN_COSMORS_24A_SOLVATION_PARAMETERS
    ),
) -> OpenCOSMORS24aSolvationResult:
    """Return the complete published open24a neutral solvation-energy ledger."""

    torch = _torch()
    if (
        isinstance(ring_atom_count, bool)
        or not isinstance(ring_atom_count, int)
        or ring_atom_count < 0
    ):
        raise ValueError("ring_atom_count must be a nonnegative integer.")
    temperature = float(temperature_k)
    if not math.isclose(temperature, 298.15, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            "The published openCOSMO-RS 24a solvation model is fixed at 298.15 K."
        )
    ionic = abs(float(solute_profile.molecular_charge_e.detach())) > 2.0e-8
    if ionic and acknowledge_unvalidated_ions is not True:
        raise ValueError(
            "openCOSMO-RS 24a was validated for neutral molecules; charged "
            "profiles require acknowledge_unvalidated_ions=true."
        )
    activity = infinite_dilution_activity(
        solute_profile,
        solvent_profile,
        temperature_k=temperature,
        discretize=discretize,
        ionic_es_solvent_class=ionic_es_solvent_class,
    )
    reference = solute_profile.dielectric_energy_hartree
    if total_solvated_minus_gas_hartree is None:
        if solute_profile.dielectric_energy_role != "total-solvated-minus-gas":
            raise ValueError(
                "A complete open24a G_solv requires total solvated-minus-gas "
                "electronic energy, not the boundary polarization term alone."
            )
        electronic_difference = reference
    else:
        electronic_difference = (
            total_solvated_minus_gas_hartree
            if isinstance(total_solvated_minus_gas_hartree, torch.Tensor)
            else reference.new_tensor(total_solvated_minus_gas_hartree)
        )
        electronic_difference = electronic_difference.to(
            dtype=torch.float64, device=reference.device
        )
        if electronic_difference.ndim != 0 or not bool(
            torch.isfinite(electronic_difference).detach()
        ):
            raise ValueError("total_solvated_minus_gas_hartree must be finite.")
    dielectric = electronic_difference * HARTREE_TO_KCAL_PER_MOL
    rt_kcal = GAS_CONSTANT_J_PER_MOL_K * temperature * JOULE_TO_KCAL
    chemical_potential = activity.total_log_activity * rt_kcal
    element_surface = _element_surface_energy(solute_profile, parameters)
    ring = reference.new_tensor(parameters.ring_atom_kcal_mol * ring_atom_count)
    eta = reference.new_tensor(parameters.eta_kcal_mol)
    if solvent_liquid_molar_volume_cm3_mol is None:
        liquid_volume = estimate_open24a_liquid_molar_volume_cm3_mol(solvent_profile)
    else:
        liquid_volume = (
            solvent_liquid_molar_volume_cm3_mol
            if isinstance(solvent_liquid_molar_volume_cm3_mol, torch.Tensor)
            else reference.new_tensor(solvent_liquid_molar_volume_cm3_mol)
        )
        liquid_volume = liquid_volume.to(dtype=torch.float64, device=reference.device)
        if (
            liquid_volume.ndim != 0
            or not bool(torch.isfinite(liquid_volume).detach())
            or float(liquid_volume.detach()) <= 0.0
        ):
            raise ValueError("solvent_liquid_molar_volume_cm3_mol must be positive.")
    ideal_gas_molar_volume_m3 = (
        GAS_CONSTANT_J_PER_MOL_K * temperature / parameters.reference_pressure_pa
    )
    reference_state = reference.new_tensor(rt_kcal) * torch.log(
        reference.new_tensor(ideal_gas_molar_volume_m3) / (liquid_volume * 1.0e-6)
    )
    total = (
        dielectric + chemical_potential - element_surface - ring - reference_state - eta
    )
    if not bool(torch.isfinite(total).detach()):
        raise FloatingPointError("open24a solvation free energy is non-finite.")
    return OpenCOSMORS24aSolvationResult(
        delta_g_solvation_kcal_mol=total,
        dielectric_kcal_mol=dielectric,
        chemical_potential_kcal_mol=chemical_potential,
        element_surface_kcal_mol=element_surface,
        ring_kcal_mol=ring,
        reference_state_kcal_mol=reference_state,
        eta_kcal_mol=eta,
        solvent_liquid_molar_volume_cm3_mol=liquid_volume,
        activity=activity,
        ionic_parameterization_validated=False if ionic else True,
        parameterization_identity=(
            "hybrid-openCOSMO-RS-24a-neutral-ORCA6"
            f"+{POLYATOMIC_ANION_SHORT_RANGE_IDENTITY}"
            if ionic_es_solvent_class is not None
            else "openCOSMO-RS-24a-neutral-ORCA6"
        ),
    )


__all__ = [
    "HARTREE_TO_KCAL_PER_MOL",
    "InfiniteDilutionActivity",
    "OPEN_COSMORS_24A_SOLVATION_PARAMETERS",
    "OpenCOSMORS24aSolvationParameters",
    "OpenCOSMORS24aSolvationResult",
    "estimate_open24a_liquid_molar_volume_cm3_mol",
    "infinite_dilution_activity",
    "open24a_solvation_free_energy",
    "staverman_guggenheim_infinite_dilution",
]

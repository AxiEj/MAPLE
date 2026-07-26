"""OpenMM-backed Amber GB models used as additive solvent corrections."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .common import EV_PER_HARTREE, KJ_PER_MOL_PER_HARTREE, build_openmm_topology
from .nonpolar import OpenMMNonpolarProvider
from .openmm_compat import (
    customgbforces_module,
    openmm_version,
    private_api_provenance,
    require_verified_openmm,
)
from .radii import GB_MODELS, OpenMMAmberGBRadiusProvider
from .result import SolvationResult

FORCE_KJMOL_NM_PER_HARTREE_ANGSTROM = KJ_PER_MOL_PER_HARTREE * 10.0
NONPOLAR_SCALE_PARAMETER = "maple_nonpolar_scale"
DEFAULT_OPENMM_PLATFORM = "CPU"
DEFAULT_CPU_PLATFORM_PROPERTIES = {
    "Threads": "1",
    "DeterministicForces": "true",
}


def _tag_unique_added_energy_term(force, baseline_force, parameter_name):
    """Scale one extra linear term and expose its energy as a derivative."""
    baseline_terms = Counter(
        (str(expression), int(computation_type))
        for expression, computation_type in (
            baseline_force.getEnergyTermParameters(index)
            for index in range(baseline_force.getNumEnergyTerms())
        )
    )
    force_terms = [
        force.getEnergyTermParameters(index)
        for index in range(force.getNumEnergyTerms())
    ]
    added_terms = (
        Counter(
            (str(expression), int(computation_type))
            for expression, computation_type in force_terms
        )
        - baseline_terms
    )
    if sum(added_terms.values()) != 1:
        raise RuntimeError(
            "OpenMM ACE decomposition expected exactly one added energy term."
        )
    added_term = next(iter(added_terms))
    indices = [
        index
        for index, (expression, computation_type) in enumerate(force_terms)
        if (str(expression), int(computation_type)) == added_term
    ]
    if len(indices) != 1:
        raise RuntimeError("OpenMM ACE energy term is not uniquely identifiable.")

    index = indices[0]
    expression, computation_type = force_terms[index]
    value, separator, definitions = str(expression).partition(";")
    scaled_expression = f"{parameter_name}*({value})"
    if separator:
        scaled_expression += f";{definitions}"
    force.addGlobalParameter(parameter_name, 1.0)
    force.addEnergyParameterDerivative(parameter_name)
    force.setEnergyTermParameters(index, scaled_expression, computation_type)


class _ContextBundle:
    def __init__(
        self,
        atoms,
        topology,
        charges,
        force_class,
        provider_parameters,
        nonpolar_provider,
        platform_name,
    ):
        from openmm import Context, Platform, System, VerletIntegrator, unit

        customgbforces = customgbforces_module()
        force_cls = getattr(customgbforces, force_class)
        sa = None if nonpolar_provider is None else nonpolar_provider.custom_gb_sa
        force = force_cls(
            solventDielectric=78.5,
            soluteDielectric=1.0,
            SA=sa,
        )
        self.nonpolar_parameter = None
        polar_force = None
        if sa == "ACE":
            polar_force = force_cls(
                solventDielectric=78.5,
                soluteDielectric=1.0,
                SA=None,
            )
        standard = np.asarray(provider_parameters, dtype=np.float64)
        if len(standard) != len(atoms):
            raise ValueError(
                "OpenMM GB parameter count does not match the MOL2 atom count."
            )
        for charge, parameters in zip(charges, standard):
            particle = [float(charge), *parameters.tolist()]
            force.addParticle(particle)
            if polar_force is not None:
                polar_force.addParticle(particle)
        force.finalize()
        if polar_force is not None:
            polar_force.finalize()
            _tag_unique_added_energy_term(
                force,
                polar_force,
                NONPOLAR_SCALE_PARAMETER,
            )
            self.nonpolar_parameter = NONPOLAR_SCALE_PARAMETER

        system = System()
        for atom in topology.atoms():
            system.addParticle(atom.element.mass)
        system.addForce(force)
        self.nonpolar_installation = None
        if nonpolar_provider is not None:
            self.nonpolar_installation = nonpolar_provider.install_separate_force(
                system,
                topology,
                atoms,
            )
        integrator = VerletIntegrator(
            0.001 * unit.picoseconds  # pyright: ignore[reportAttributeAccessIssue]
        )
        available = [
            Platform.getPlatform(index).getName()
            for index in range(Platform.getNumPlatforms())
        ]
        match = next(
            (name for name in available if name.lower() == str(platform_name).lower()),
            None,
        )
        if match is None:
            raise ValueError(
                f"Unknown OpenMM platform {platform_name!r}; available: {', '.join(available)}."
            )
        platform = Platform.getPlatformByName(match)
        platform_properties = (
            dict(DEFAULT_CPU_PLATFORM_PROPERTIES)
            if platform.getName().lower() == "cpu"
            else {}
        )
        context = Context(system, integrator, platform, platform_properties)
        self.force = force
        self.context = context
        self.integrator = integrator
        self.platform_name = platform.getName()
        self.platform_properties = {
            name: platform.getPropertyValue(context, name)
            for name in sorted(platform_properties)
        }

    def set_charges(self, charges):
        for index, charge in enumerate(charges):
            params = list(self.force.getParticleParameters(index))
            params[0] = float(charge)
            self.force.setParticleParameters(index, params)
        self.force.updateParametersInContext(self.context)

    def evaluate(self, positions_angstrom, need_forces):
        from openmm import unit

        self.context.setPositions(
            np.asarray(positions_angstrom)
            * 0.1
            * unit.nanometer  # pyright: ignore[reportAttributeAccessIssue]
        )
        state = self.context.getState(
            getEnergy=True,
            getForces=need_forces,
            getParameterDerivatives=self.nonpolar_parameter is not None,
        )
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = None
        if need_forces:
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole
                / unit.nanometer  # pyright: ignore[reportAttributeAccessIssue]
            )
        nonpolar_energy = None
        if self.nonpolar_parameter is not None:
            derivatives = state.getEnergyParameterDerivatives()
            nonpolar_energy = float(derivatives[self.nonpolar_parameter])
        return (
            float(energy),
            None if forces is None else np.asarray(forces, dtype=np.float64),
            nonpolar_energy,
        )


class OpenMMGB:
    """HCT/OBC-I/OBC-II/GBn/GBn2 correction with upstream OpenMM forces."""

    supported_properties = frozenset({"energy", "forces"})

    def __init__(
        self,
        atoms,
        charges,
        *,
        model: str = "obc2",
        nonpolar: str = "ace",
        platform: str = DEFAULT_OPENMM_PLATFORM,
    ):
        require_verified_openmm()
        model = str(model).lower()
        nonpolar = str(nonpolar).lower()
        if model not in GB_MODELS:
            raise ValueError(
                f"Unknown GB model {model!r}; choose hct, obc1, obc2, gbn, or gbn2."
            )
        if model == "gbn2":
            symbols = set(atoms.get_chemical_symbols())
            if "P" in symbols:
                raise NotImplementedError(
                    "OpenMM's generic-molecule GBn2 parameter table does not provide "
                    "Amber's phosphorus-specific alpha/beta/gamma parameters. MAPLE "
                    "refuses the upstream default fallback for phosphorus until an "
                    "authoritative OpenMM implementation passes Amber parity."
                )
            if "S" in symbols:
                raise NotImplementedError(
                    "OpenMM's generic GBn2 expression does not reproduce Amber's signed "
                    "near-pair descreening branch for sulfur's negative screening radius. "
                    "MAPLE refuses sulfur-containing GBn2 until an authoritative "
                    "implementation passes Amber energy/force parity."
                )
        self.model = model
        self.nonpolar = nonpolar
        self.topology = build_openmm_topology(atoms)
        self.radius_provider = OpenMMAmberGBRadiusProvider(model)
        self.radius_result = self.radius_provider.assign(self.topology)
        self.nonpolar_provider = OpenMMNonpolarProvider(nonpolar)
        self.charges = np.asarray(charges, dtype=np.float64).copy()
        if self.charges.shape != (len(atoms),) or not np.isfinite(self.charges).all():
            raise ValueError("OpenMM GB requires one finite partial charge per atom.")
        info = GB_MODELS[model]
        context_args = (
            atoms,
            self.topology,
            self.charges,
            info["class"],
            self.radius_result.provider_parameters,
        )
        self._polar = _ContextBundle(*context_args, None, platform)
        self._total = (
            self._polar
            if not self.nonpolar_provider.enabled
            else _ContextBundle(*context_args, self.nonpolar_provider, platform)
        )
        self.platform = self._total.platform_name
        self._provenance = {
            "provider": "openmm",
            "provider_version": openmm_version(),
            "private_api_compatibility": private_api_provenance(),
            "method": "gb",
            "platform": self._total.platform_name,
            "platform_properties": self._total.platform_properties,
            "model": model,
            "amber_igb": info["igb"],
            "radii": info["radii"],
            "profile": info["profile"],
            "nonpolar": nonpolar,
            "nonpolar_implementation": self.nonpolar_provider.provenance[
                "implementation"
            ],
            "component_decomposition": {
                "ace": "single-context OpenMM energy-parameter derivative",
                "lcpo": "polar/total OpenMM context energy difference",
                "none": "nonpolar component disabled",
            }[nonpolar],
            "energy_force_evaluations_per_call": 2 if nonpolar == "lcpo" else 1,
            "radius_provider": self.radius_result.provenance,
            "nonpolar_provider": self.nonpolar_provider.provenance,
            "nonpolar_installation": self._total.nonpolar_installation,
            "solvent": "water",
            "solvent_dielectric": 78.5,
            "solute_dielectric": 1.0,
        }

    @property
    def provenance(self):
        return dict(self._provenance)

    def _set_charges(self, charges):
        charges = np.asarray(charges, dtype=np.float64)
        if charges.shape != self.charges.shape or not np.isfinite(charges).all():
            raise ValueError("OpenMM GB requires one finite partial charge per atom.")
        self._polar.set_charges(charges)
        if self._total is not self._polar:
            self._total.set_charges(charges)
        self.charges = charges.copy()

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        charges=None,
        calculator=None,
    ) -> SolvationResult:
        if charges is not None and not np.array_equal(
            np.asarray(charges), self.charges
        ):
            self._set_charges(charges)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        total_kj, total_force, nonpolar_kj = self._total.evaluate(
            positions,
            need_forces,
        )
        if self._total is self._polar:
            polar_kj = total_kj
            nonpolar_kj = 0.0
        elif nonpolar_kj is not None:
            polar_kj = total_kj - nonpolar_kj
        else:
            polar_kj = self._polar.evaluate(positions, False)[0]
            nonpolar_kj = total_kj - polar_kj
        forces = None
        if total_force is not None:
            forces = total_force / FORCE_KJMOL_NM_PER_HARTREE_ANGSTROM
        return SolvationResult(
            energy_hartree=total_kj / KJ_PER_MOL_PER_HARTREE,
            forces_hartree_per_angstrom=forces,
            components_hartree={
                "polar": polar_kj / KJ_PER_MOL_PER_HARTREE,
                "nonpolar": nonpolar_kj / KJ_PER_MOL_PER_HARTREE,
            },
            provenance=self.provenance,
        )

    def polar_charge_hessian_ev(self, atoms) -> np.ndarray:
        """Return K where G_GB,polar(q)=0.5*q.T@K@q, in eV/e^2."""
        n = len(atoms)
        original_charges = self.charges.copy()
        zero = np.zeros(n, dtype=np.float64)
        try:
            self._set_charges(zero)
            baseline = self._polar.evaluate(atoms.get_positions(), False)[0]
            if abs(baseline) > 1.0e-9:
                raise ValueError("OpenMM polar GB energy is not zero at zero charge.")
            diagonal_energies = np.zeros(n, dtype=np.float64)
            matrix = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                q = np.zeros(n)
                q[i] = 1.0
                self._set_charges(q)
                diagonal_energies[i] = self._polar.evaluate(
                    atoms.get_positions(), False
                )[0]
                matrix[i, i] = 2.0 * diagonal_energies[i]
            for i in range(n):
                for j in range(i + 1, n):
                    q = np.zeros(n)
                    q[i] = q[j] = 1.0
                    self._set_charges(q)
                    pair_energy = self._polar.evaluate(atoms.get_positions(), False)[0]
                    matrix[i, j] = matrix[j, i] = (
                        pair_energy - diagonal_energies[i] - diagonal_energies[j]
                    )
        finally:
            self._set_charges(original_charges)
        return matrix * (EV_PER_HARTREE / KJ_PER_MOL_PER_HARTREE)

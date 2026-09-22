"""OpenMM-backed Amber GB models used as additive solvent corrections."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .common import KJ_PER_MOL_PER_HARTREE, build_openmm_topology
from .nonpolar import OpenMMNonpolarProvider
from .obc2_parameters import build_obc2_parameters
from .openmm_execution import CPU_PROPERTIES, resolve_openmm_platform
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
DEFAULT_CPU_PLATFORM_PROPERTIES = CPU_PROPERTIES


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
        *,
        model_device=None,
        precision=None,
        device_index=None,
        opencl_platform_index=None,
    ):
        from openmm import Context, System, VerletIntegrator, unit

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
        platform, platform_properties, self.execution = resolve_openmm_platform(
            platform_name, model_device=model_device, precision=precision,
            device_index=device_index, opencl_platform_index=opencl_platform_index,
        )
        context = Context(system, integrator, platform, platform_properties)
        self.force = force
        self.context = context
        self.integrator = integrator
        self.platform_name = platform.getName()
        self.platform_properties = {
            name: platform.getPropertyValue(context, name)
            for name in sorted(platform.getPropertyNames())
        }
        self.execution["effective_properties"] = dict(self.platform_properties)

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
        model_device=None,
        precision: str | None = None,
        device_index=None,
        opencl_platform_index=None,
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
        self.obc2_parameters = (
            build_obc2_parameters(
                self.charges,
                self.radius_result,
                nonpolar=nonpolar,
            )
            if model == "obc2" and nonpolar in {"ace", "none"}
            else None
        )
        self._derivative_device = (
            "cpu" if model_device is None else str(model_device)
        )
        self._derivative_backend = None
        shared_provider_parameters = (
            self.obc2_parameters.provider_parameters
            if self.obc2_parameters is not None
            else self.radius_result.provider_parameters
        )
        context_args = (
            atoms,
            self.topology,
            self.charges,
            info["class"],
            shared_provider_parameters,
        )
        execution_options = dict(model_device=model_device, precision=precision,
                                 device_index=device_index, opencl_platform_index=opencl_platform_index)
        self._polar = _ContextBundle(*context_args, None, platform, **execution_options)
        self._total = (
            self._polar
            if not self.nonpolar_provider.enabled
            else _ContextBundle(*context_args, self.nonpolar_provider, platform, **execution_options)
        )
        self.platform = self._total.platform_name
        self._provenance = {
            "provider": "openmm",
            "provider_version": openmm_version(),
            "private_api_compatibility": private_api_provenance(),
            "method": "gb",
            "platform": self._total.platform_name,
            "platform_properties": self._total.platform_properties,
            "execution": self._total.execution,
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
            "analytic_derivative_profile": (
                "torch-obc2-v1"
                if self.obc2_parameters is not None
                else None
            ),
        }

    @property
    def provenance(self):
        return dict(self._provenance)

    def derivative_backend(self):
        """Lazily construct the parity-gated Torch OBC-II derivative kernel."""
        if self.obc2_parameters is None:
            raise NotImplementedError(
                "Analytic solvent derivatives are available only for OBC-II "
                "with nonpolar=ace or nonpolar=none."
            )
        if self._derivative_backend is None:
            from .torch_obc2 import TorchOBC2

            self._derivative_backend = TorchOBC2(
                self.obc2_parameters,
                device=self._derivative_device,
            )
        return self._derivative_backend

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
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

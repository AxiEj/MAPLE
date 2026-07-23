"""OpenMM-backed Amber GB models used as additive solvent corrections."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import numpy as np

from .result import SolvationResult


KJ_PER_MOL_PER_HARTREE = 2625.4996394799
EV_PER_HARTREE = 27.211386245988
FORCE_KJMOL_NM_PER_HARTREE_ANGSTROM = KJ_PER_MOL_PER_HARTREE * 10.0

GB_MODELS: dict[str, dict[str, Any]] = {
    "hct": {"class": "GBSAHCTForce", "igb": 1, "radii": "mbondi", "profile": "hct-mbondi"},
    "obc1": {"class": "GBSAOBC1Force", "igb": 2, "radii": "mbondi2", "profile": "obc1-mbondi2"},
    "obc2": {"class": "GBSAOBC2Force", "igb": 5, "radii": "mbondi2", "profile": "obc2-mbondi2"},
    "gbn": {"class": "GBSAGBnForce", "igb": 7, "radii": "bondi", "profile": "gbn-bondi"},
    "gbn2": {"class": "GBSAGBn2Force", "igb": 8, "radii": "mbondi3", "profile": "gbn2-mbondi3"},
}


def build_openmm_topology(atoms):
    try:
        from openmm import app
    except ImportError as exc:
        raise ImportError(
            "OpenMM GB requires the optional dependency. Install with "
            "`pip install 'maple[implicit-gb]'`."
        ) from exc

    metadata = atoms.info.get("mol2")
    if not metadata:
        raise ValueError("OpenMM GB requires MOL2 topology metadata.")
    topology = app.Topology()
    chain = topology.addChain("A")
    # MOL2 molecule names are identifiers, not biomolecular residue types.
    # A synthetic name prevents an arbitrary input name such as "DA" from
    # selecting GBn2's nucleic-acid-specific parameter branch.
    residue = topology.addResidue("MOL", chain)
    omm_atoms = []
    names = metadata.get("atom_names") or atoms.get_chemical_symbols()
    for name, symbol in zip(names, atoms.get_chemical_symbols()):
        omm_atoms.append(topology.addAtom(str(name), app.Element.getBySymbol(symbol), residue))
    for i, j, _bond_type in metadata.get("bonds", []):
        topology.addBond(omm_atoms[int(i)], omm_atoms[int(j)])
    return topology


class _ContextBundle:
    def __init__(self, atoms, topology, charges, model, sa, platform_name):
        from openmm import Context, Platform, System, VerletIntegrator, unit
        from openmm.app.internal import customgbforces

        model_info = GB_MODELS[model]
        force_cls = getattr(customgbforces, model_info["class"])
        lcpo_requested = str(sa).upper() == "LCPO"
        try:
            force = force_cls(
                solventDielectric=78.5,
                soluteDielectric=1.0,
                SA=None if lcpo_requested else sa,
            )
        except ValueError as exc:
            if lcpo_requested:
                raise ImportError(
                    "nonpolar=lcpo requires an OpenMM release whose CustomGB forces expose LCPO "
                    "(MAPLE targets OpenMM>=8.5)."
                ) from exc
            raise
        standard = np.asarray(force_cls.getStandardParameters(topology), dtype=np.float64)
        if len(standard) != len(atoms):
            raise ValueError("OpenMM GB parameter count does not match the MOL2 atom count.")
        for charge, parameters in zip(charges, standard):
            force.addParticle([float(charge), *parameters.tolist()])
        force.finalize()

        system = System()
        for atom in topology.atoms():
            system.addParticle(atom.element.mass)
        system.addForce(force)
        if lcpo_requested:
            try:
                from openmm.app.internal import lcpo

                lcpo.addLCPOForce(
                    system,
                    lcpo.getLCPOParamsTopology(topology),
                    usePeriodic=False,
                )
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "nonpolar=lcpo requires OpenMM>=8.5 with LCPOForce support."
                ) from exc
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
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
        context = Context(system, integrator, platform)
        self.force = force
        self.context = context
        self.integrator = integrator

    def set_charges(self, charges):
        for index, charge in enumerate(charges):
            params = list(self.force.getParticleParameters(index))
            params[0] = float(charge)
            self.force.setParticleParameters(index, params)
        self.force.updateParametersInContext(self.context)

    def evaluate(self, positions_angstrom, need_forces):
        from openmm import unit

        self.context.setPositions(np.asarray(positions_angstrom) * 0.1 * unit.nanometer)
        state = self.context.getState(getEnergy=True, getForces=need_forces)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = None
        if need_forces:
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        return float(energy), None if forces is None else np.asarray(forces, dtype=np.float64)


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
        platform: str = "Reference",
    ):
        model = str(model).lower()
        nonpolar = str(nonpolar).lower()
        if model not in GB_MODELS:
            raise ValueError(f"Unknown GB model {model!r}; choose hct, obc1, obc2, gbn, or gbn2.")
        if nonpolar not in {"ace", "lcpo", "none"}:
            raise ValueError("OpenMM GB nonpolar must be ace, lcpo, or none.")
        if model == "gbn2" and "P" in atoms.get_chemical_symbols():
            raise NotImplementedError(
                "OpenMM's generic-molecule GBn2 parameter table does not provide Amber's "
                "phosphorus-specific alpha/beta/gamma parameters. MAPLE refuses the upstream "
                "default fallback for phosphorus until an authoritative OpenMM implementation "
                "passes Amber parity."
            )
        self.model = model
        self.nonpolar = nonpolar
        self.platform = platform
        self.topology = build_openmm_topology(atoms)
        self.charges = np.asarray(charges, dtype=np.float64).copy()
        if self.charges.shape != (len(atoms),) or not np.isfinite(self.charges).all():
            raise ValueError("OpenMM GB requires one finite partial charge per atom.")
        sa = None if nonpolar == "none" else nonpolar.upper()
        self._polar = _ContextBundle(atoms, self.topology, self.charges, model, None, platform)
        self._total = self._polar if sa is None else _ContextBundle(
            atoms, self.topology, self.charges, model, sa, platform
        )
        try:
            version = importlib.metadata.version("openmm")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        info = GB_MODELS[model]
        nonpolar_implementation = {
            "none": "disabled",
            "ace": "openmm.app.internal.customgbforces CustomGBForce ACE term",
            "lcpo": "openmm.app.internal.lcpo.LCPOForce",
        }[nonpolar]
        self._provenance = {
            "provider": "openmm",
            "provider_version": version,
            "method": "gb",
            "model": model,
            "amber_igb": info["igb"],
            "radii": info["radii"],
            "profile": info["profile"],
            "nonpolar": nonpolar,
            "nonpolar_implementation": nonpolar_implementation,
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
        if charges is not None and not np.array_equal(np.asarray(charges), self.charges):
            self._set_charges(charges)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        polar_kj, polar_force = self._polar.evaluate(positions, need_forces)
        if self._total is self._polar:
            total_kj, total_force = polar_kj, polar_force
        else:
            total_kj, total_force = self._total.evaluate(positions, need_forces)
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
                diagonal_energies[i] = self._polar.evaluate(atoms.get_positions(), False)[0]
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

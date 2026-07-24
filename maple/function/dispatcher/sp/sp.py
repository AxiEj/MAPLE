from typing import Iterator, List, Optional, Union
from dataclasses import dataclass

from ase import Atoms

from ...calculator._batch_eval import (
    EnergyEvaluator,
    PathEvaluator,
    shared_calculator,
    structures_have_constraints,
    supports_batch_calculation,
)
from ..jobABC import JobABC
from maple.function.timer import timer

@dataclass
class SPParams:
    """Parameters for Single Point calculation."""
    verbose: int = 0  # 0=coordinates+energy+charge/mult, 1=+gradients

class SinglePoint(JobABC):

    def __init__(self, output: str, atoms: Union[Atoms, List[Atoms]],
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.is_trajectory = isinstance(atoms, list)

        # Initialize params
        self.params = self._init_params(SPParams, paras, ("sp", "SP"))
        self.verbose = self.params.verbose

    def run(self):
        if self.is_trajectory:
            self._run_trajectory()
        else:
            self._run_single()

    def _run_single(self):
        """Original single-point calculation logic."""
        with timer("Single Point Energy Calculation"):
            energy = self.atoms.get_potential_energy()
            self.log_info(self._single_energy_lines(energy))

    def _single_energy_lines(self, energy: float) -> list:
        """Return single-structure SP result lines for the selected verbosity."""
        lines = ["\n"]
        lines.extend(self._charge_mult_lines(self.atoms))
        lines.append(f"Energy: {energy:.10f} Hartree\n")
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(self.atoms))
        return lines

    def _charge_mult_lines(self, atoms: Atoms) -> list:
        """Return charge and multiplicity metadata lines for SP output."""
        charge = atoms.info.get('charge', 0)
        mult = atoms.info.get('mult', 1)
        return [f"Charge: {charge}, Multiplicity: {mult}\n"]

    def _gradient_lines(self, atoms: Atoms, forces=None) -> list:
        """Return per-atom energy gradients for detailed SP output."""
        if forces is None:
            forces = atoms.get_forces()
        gradients = -forces
        symbols = atoms.get_chemical_symbols()
        lines = [
            "\nGradients (Hartree/Angstrom):\n",
            "  Gradient = -Force\n",
            "  Atom  El"
            "        Gx              Gy              Gz\n",
        ]
        for i, (sym, gradient) in enumerate(zip(symbols, gradients), start=1):
            lines.append(
                f"  {i:<4} {sym:<2}"
                f" {gradient[0]:>15.8f} {gradient[1]:>15.8f} {gradient[2]:>15.8f}\n"
            )
        return lines

    def _trajectory_frame_lines(
        self,
        idx: int,
        atoms_frame: Atoms,
        energy_hartree: float,
        forces=None,
    ) -> list:
        """Return trajectory-frame SP result lines for the selected verbosity."""
        lines = [
            f"\n{('Frame ' + str(idx)):=^80}\n",
        ]
        lines.extend(self._charge_mult_lines(atoms_frame))
        lines.extend([
            f"Energy: {energy_hartree:.10f} Hartree\n\n",
            "Coordinates (Angstrom):\n",
        ])
        symbols = atoms_frame.get_chemical_symbols()
        positions = atoms_frame.get_positions()
        for i, (sym, pos) in enumerate(zip(symbols, positions), start=1):
            lines.append(f"  {i:<4} {sym:<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}\n")
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(atoms_frame, forces=forces))
        lines.append("=" * 80 + "\n")
        return lines

    def _trajectory_energy_forces(self):
        """Evaluate trajectory frames with a batch path when it is safe.

        SP trajectories are independent structures, so they are a natural batch
        workload.  Mixed/no calculator inputs keep the original sequential ASE
        behavior.
        """
        calc = shared_calculator(self.atoms)
        if (
            calc is None
            or not supports_batch_calculation(calc)
            or structures_have_constraints(self.atoms)
        ):
            energies = []
            forces = [] if self.verbose >= 1 else None
            for atoms_frame in self.atoms:
                energies.append(float(atoms_frame.get_potential_energy()))
                if forces is not None:
                    forces.append(atoms_frame.get_forces())
            return energies, forces

        if self.verbose >= 1:
            energies, forces = PathEvaluator(
                calc,
                batch_size=getattr(calc, "path_batch_size", None),
            ).energy_forces(self.atoms)
            return [float(e) for e in energies], forces

        energies = EnergyEvaluator(
            calc,
            batch_size=getattr(calc, "path_batch_size", None),
        ).energies(self.atoms)
        return [float(e) for e in energies], None

    def _run_trajectory(self):
        """Process multiple independent structures."""
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(self.atoms)
            self.log_info([
                f"\nProcessing {n_frames} structures from trajectory...\n",
                "=" * 80 + "\n",
            ])

            energies_hartree, forces_list = self._trajectory_energy_forces()
            self.log_info(
                self._trajectory_result_lines(energies_hartree, forces_list)
            )

    def _trajectory_result_lines(
        self,
        energies_hartree: List[float],
        forces_list=None,
    ) -> Iterator[str]:
        """Yield trajectory results in output order for one streamed write."""
        for idx, (atoms_frame, energy_hartree) in enumerate(
            zip(self.atoms, energies_hartree),
            start=1,
        ):
            forces = None if forces_list is None else forces_list[idx - 1]
            yield from self._trajectory_frame_lines(
                idx,
                atoms_frame,
                energy_hartree,
                forces=forces,
            )

        yield f"\n{' SUMMARY ':=^80}\n"
        yield f"Total frames processed: {len(self.atoms)}\n"
        energy_min = min(energies_hartree)
        energy_max = max(energies_hartree)
        yield (
            f"Energy range: {energy_min:.10f} to "
            f"{energy_max:.10f} Hartree\n"
        )
        energy_span = energy_max - energy_min
        yield (
            f"Energy span: {energy_span:.10f} Hartree "
            f"({energy_span * 627.509:.4f} kcal/mol)\n"
        )
        yield "=" * 80 + "\n"

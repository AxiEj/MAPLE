from dataclasses import dataclass

import numpy as np
from ase import Atoms

from maple.function.timer import timer

from ..jobABC import JobABC


@dataclass
class SPParams:
    """Parameters for Single Point calculation."""

    verbose: int = 0  # 0=coordinates+energy+charge/mult, 1=+gradients


class SinglePoint(JobABC):
    def __init__(
        self,
        output: str,
        atoms: Atoms | list[Atoms],
        paras: dict | None = None,
    ):
        super().__init__(output)
        self.atoms = atoms
        self.is_trajectory = isinstance(atoms, list)

        # Initialize params
        self.params = self._init_params(SPParams, paras, ("sp", "SP"))
        self.verbose = self.params.verbose

    def run(self):
        # ``SinglePoint`` is also a public direct job API in existing MAPLE
        # integrations, so it owns the same idempotent unit boundary as the
        # Dispatcher.  If Dispatcher already installed a view, the context
        # manager recognizes it and performs no second conversion.
        from ..legacy_units import legacy_hartree_job_calculators

        with legacy_hartree_job_calculators(self.atoms):
            self._run_legacy()

    def _run_legacy(self):
        if self.is_trajectory:
            self._run_trajectory()
        else:
            self._run_single()

    def _run_single(self):
        """Original single-point calculation logic."""
        with timer("Single Point Energy Calculation"):
            energy, forces = self._evaluate_structure(self.atoms)
            self.log_info(self._single_energy_lines(energy, forces=forces))

    def _evaluate_structure(
        self,
        atoms: Atoms,
    ) -> tuple[float, np.ndarray | None]:
        """Evaluate exactly the property set needed by the selected output.

        ASE force calculations conventionally populate energy and forces in
        one calculator call.  Requesting forces first therefore avoids an
        energy-only Route-2 solve followed by a second force-capable solve.
        """

        forces = None
        if self.verbose >= 1:
            forces = np.asarray(atoms.get_forces(), dtype=float)
        energy = float(atoms.get_potential_energy())
        return energy, forces

    def _single_energy_lines(
        self,
        energy: float,
        *,
        forces: np.ndarray | None = None,
    ) -> list[str]:
        """Return single-structure SP result lines for the selected verbosity."""
        lines = ["\n"]
        lines.extend(self._charge_mult_lines(self.atoms))
        lines.append(f"Energy: {energy:.10f} Hartree\n")
        lines.extend(self._solvation_lines(self.atoms))
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(self.atoms, forces=forces))
        return lines

    @staticmethod
    def _solvation_lines(atoms: Atoms) -> list[str]:
        results = getattr(getattr(atoms, "calc", None), "results", {})
        solvation = results.get("solvation") if isinstance(results, dict) else None
        if not isinstance(solvation, dict):
            return []
        gas = solvation.get("gas_energy_hartree")
        delta = solvation.get(
            "delta_g_solv_hartree", solvation.get("energy_hartree")
        )
        combined = solvation.get("combined_energy_hartree")
        if gas is None or delta is None or combined is None:
            return []
        provenance = solvation.get("provenance", {})
        standard_state = provenance.get("standard_state", "provider-defined")
        return [
            f"Gas-phase MLIP energy: {float(gas):.10f} Hartree\n",
            (
                "Solvation free-energy correction (Delta G_solv, "
                f"{standard_state}): {float(delta):.10f} Hartree\n"
            ),
            (
                "Combined E_MLIP(gas)+Delta G_solv: "
                f"{float(combined):.10f} Hartree\n"
            ),
            (
                "ASE free_energy is the combined electronic-plus-solvation "
                "value, not a thermochemical Gibbs free energy.\n"
            ),
        ]

    def _charge_mult_lines(self, atoms: Atoms) -> list[str]:
        """Return charge and multiplicity metadata lines for SP output."""
        charge = atoms.info.get("charge", 0)
        mult = atoms.info.get("mult", 1)
        return [f"Charge: {charge}, Multiplicity: {mult}\n"]

    def _gradient_lines(
        self,
        atoms: Atoms,
        *,
        forces: np.ndarray | None = None,
    ) -> list[str]:
        """Return per-atom energy gradients for detailed SP output."""
        evaluated_forces = (
            np.asarray(atoms.get_forces(), dtype=float)
            if forces is None
            else np.asarray(forces, dtype=float)
        )
        if evaluated_forces.shape != (len(atoms), 3):
            raise ValueError("Single-point forces must have shape (n_atoms, 3).")
        gradients = -evaluated_forces
        symbols = atoms.get_chemical_symbols()
        lines = [
            "\nGradients (Hartree/Angstrom):\n",
            "  Gradient = -Force\n",
            "  Atom  El        Gx              Gy              Gz\n",
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
        *,
        forces: np.ndarray | None = None,
    ) -> list[str]:
        """Return trajectory-frame SP result lines for the selected verbosity."""
        lines = [
            f"\n{('Frame ' + str(idx)):=^80}\n",
        ]
        lines.extend(self._charge_mult_lines(atoms_frame))
        lines.extend(
            [
                f"Energy: {energy_hartree:.10f} Hartree\n\n",
                "Coordinates (Angstrom):\n",
            ]
        )
        symbols = atoms_frame.get_chemical_symbols()
        positions = atoms_frame.get_positions()
        for i, (sym, pos) in enumerate(zip(symbols, positions), start=1):
            lines.append(
                f"  {i:<4} {sym:<2} {pos[0]:>15.8f} "
                f"{pos[1]:>15.8f} {pos[2]:>15.8f}\n"
            )
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(atoms_frame, forces=forces))
        lines.append("=" * 80 + "\n")
        return lines

    def _run_trajectory(self):
        """Process multiple structures sequentially."""
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(self.atoms)
            self.log_info(
                [f"\nProcessing {n_frames} structures from trajectory...\n"]
            )
            self.log_info(["=" * 80 + "\n"])

            energies_hartree = []

            for idx, atoms_frame in enumerate(self.atoms, start=1):
                energy_hartree, forces = self._evaluate_structure(atoms_frame)
                energies_hartree.append(energy_hartree)

                self.log_info(
                    self._trajectory_frame_lines(
                        idx,
                        atoms_frame,
                        energy_hartree,
                        forces=forces,
                    )
                )

            # Summary (always shown)
            self.log_info([f"\n{' SUMMARY ':=^80}\n"])
            self.log_info([f"Total frames processed: {n_frames}\n"])
            self.log_info(
                [
                    (
                        "Energy range: "
                        f"{min(energies_hartree):.10f} to "
                        f"{max(energies_hartree):.10f} Hartree\n"
                    )
                ]
            )
            energy_span = max(energies_hartree) - min(energies_hartree)
            self.log_info(
                [
                    (
                        f"Energy span: {energy_span:.10f} Hartree "
                        f"({energy_span * 627.509:.4f} kcal/mol)\n"
                    )
                ]
            )
            self.log_info(["=" * 80 + "\n"])

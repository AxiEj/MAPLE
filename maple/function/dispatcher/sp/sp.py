from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from maple.function.timer import timer

from ..jobABC import JobABC

if TYPE_CHECKING:
    from .result import SPBatchResult, SPFrameContext, SPFrameResult


@dataclass
class SPParams:
    """Parameters for Single Point calculation."""

    verbose: int = 0  # 0=coordinates+energy+charge/mult, 1=+gradients
    mode: str = "single"


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

    def run(self) -> SPBatchResult | SPFrameResult | None:
        # ``SinglePoint`` is also a public direct job API in existing MAPLE
        # integrations, so it owns the same idempotent unit boundary as the
        # Dispatcher.  If Dispatcher already installed a view, the context
        # manager recognizes it and performs no second conversion.
        from ..legacy_units import legacy_hartree_job_calculators

        from ..pure_nonmd_status import is_pure_torch_v4
        from .result import (
            normalize_v4_sp_request,
            preflight_same_identity_frames,
            sp_request_frames,
        )

        frames = sp_request_frames(self.atoms)
        v4_flags = [is_pure_torch_v4(frame) for frame in frames]
        if any(v4_flags) and not all(v4_flags):
            raise ValueError(
                "V4 multi-frame SP requires every frame to use the same "
                "canonical V4 identity."
            )
        if v4_flags and all(v4_flags):
            request = normalize_v4_sp_request(
                self.atoms, {"mode": self.params.mode}
            )
            contexts = preflight_same_identity_frames(request.frames)
            with legacy_hartree_job_calculators(self.atoms):
                if request.returns_batch:
                    return self._run_trajectory(contexts, frames=request.frames)
                return self._run_single(contexts[0], atoms=request.frames[0])
        with legacy_hartree_job_calculators(self.atoms):
            return self._run_legacy()

    def _run_legacy(
        self, contexts: tuple[SPFrameContext, ...] | None = None
    ) -> SPBatchResult | SPFrameResult | None:
        if self.is_trajectory:
            return self._run_trajectory(contexts)
        else:
            return self._run_single(None if contexts is None else contexts[0])

    def _run_single(
        self,
        context: SPFrameContext | None = None,
        *,
        atoms: Atoms | None = None,
    ) -> SPFrameResult | None:
        """Original single-point calculation logic."""
        if atoms is None:
            if not isinstance(self.atoms, Atoms):
                raise ValueError("Single-point mode=single requires one Atoms object.")
            target = self.atoms
        else:
            target = atoms
        with timer("Single Point Energy Calculation"):
            if context is not None:
                from .result import require_unchanged_frame_context

                require_unchanged_frame_context(target, context)
            energy, forces = self._evaluate_structure(target, context=context)
            result = None
            if context is not None:
                from .result import (
                    capture_raw_frame_result,
                    require_unchanged_frame_context,
                )

                require_unchanged_frame_context(target, context)
                result = capture_raw_frame_result(
                    target, context, forces_requested=self.verbose >= 1
                )
            self.log_info(
                self._single_energy_lines(
                    energy, forces=forces, atoms=target
                )
            )
            return result

    def _evaluate_structure(
        self,
        atoms: Atoms,
        *,
        context: SPFrameContext | None = None,
    ) -> tuple[float, np.ndarray | None]:
        """Evaluate exactly the property set needed by the selected output.

        ASE force calculations conventionally populate energy and forces in
        one calculator call.  Requesting forces first therefore avoids an
        energy-only Route-2 solve followed by a second force-capable solve.
        """

        forces = None
        if self.verbose >= 1:
            forces = np.asarray(atoms.get_forces(), dtype=float)
            if context is not None:
                from .result import require_unchanged_frame_context

                require_unchanged_frame_context(atoms, context)
        energy = float(atoms.get_potential_energy())
        return energy, forces

    def _single_energy_lines(
        self,
        energy: float,
        *,
        forces: np.ndarray | None = None,
        atoms: Atoms | None = None,
    ) -> list[str]:
        """Return single-structure SP result lines for the selected verbosity."""
        if atoms is None:
            if not isinstance(self.atoms, Atoms):
                raise ValueError("Single-point mode=single requires one Atoms object.")
            target = self.atoms
        else:
            target = atoms
        lines = ["\n"]
        lines.extend(self._charge_mult_lines(target))
        lines.append(f"Energy: {energy:.10f} Hartree\n")
        lines.extend(self._solvation_lines(target))
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(target, forces=forces))
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

    def _run_trajectory(
        self,
        contexts: tuple[SPFrameContext, ...] | None = None,
        *,
        frames: tuple[Atoms, ...] | None = None,
    ) -> SPBatchResult | None:
        """Process multiple structures sequentially."""
        if frames is None:
            if not isinstance(self.atoms, list):
                raise ValueError("Legacy trajectory execution requires a frame list.")
            targets = tuple(self.atoms)
        else:
            targets = frames
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(targets)
            self.log_info(
                [f"\nProcessing {n_frames} structures from trajectory...\n"]
            )
            self.log_info(["=" * 80 + "\n"])

            energies_hartree = []
            frame_results = []

            for frame_index, atoms_frame in enumerate(targets):
                idx = frame_index + 1
                if contexts is not None:
                    from .result import require_unchanged_frame_context

                    require_unchanged_frame_context(
                        atoms_frame, contexts[frame_index]
                    )
                energy_hartree, forces = self._evaluate_structure(
                    atoms_frame,
                    context=(
                        None if contexts is None else contexts[frame_index]
                    ),
                )
                energies_hartree.append(energy_hartree)
                if contexts is not None:
                    from .result import (
                        capture_raw_frame_result,
                        require_unchanged_frame_context,
                    )

                    require_unchanged_frame_context(
                        atoms_frame, contexts[frame_index]
                    )
                    frame_results.append(
                        capture_raw_frame_result(
                            atoms_frame,
                            contexts[frame_index],
                            forces_requested=self.verbose >= 1,
                        )
                    )

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
            if contexts is not None:
                from .result import SPBatchResult

                return SPBatchResult(tuple(frame_results))
            return None
